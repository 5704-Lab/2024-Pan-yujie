"""Single-GPU independent adapter training; original trainer remains unchanged."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from manifest_dataset import ManifestDataset
from regional_adapters import (create_processors, file_digest, processors_from_bank,
                               read_bank, save_bank)
from regional_runtime import check_local_diffusers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local SD1.5 directory")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=["anchor", "expert"], required=True)
    parser.add_argument("--anchor", help="Same normal adapter_bank.pt for every expert")
    parser.add_argument("--freeze-saim", action="store_true", help="Ablation: retain anchor temporal_* parameters")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", help="training_state.pt, including optimizer and RNG")
    return parser.parse_args()


def main(args):
    if min(args.steps, args.save_every, args.batch_size, args.accumulation, args.rank) < 1:
        raise ValueError("steps, save-every, batch-size, accumulation and rank must be positive")
    if args.role == "expert" and not args.anchor:
        raise ValueError("Every expert must initialize from --anchor")
    if args.role == "anchor" and (args.anchor or args.freeze_saim or args.category != "normal"):
        raise ValueError("Anchor requires category=normal, no --anchor and no --freeze-saim")
    if not Path(args.base_model).is_dir():
        raise FileNotFoundError("Download SD1.5 to --base-model before training")
    dataset = ManifestDataset(args.manifest, args.category, args.resolution, args.role == "anchor")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; CPU routing tests do not require CUDA")
    check_local_diffusers()
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    base_model = str(Path(args.base_model).resolve())
    metadata = dict(role=args.role, category=args.category, base_model=base_model,
                    prompt_blend="a vfx with sks", prompt_fg="sks", seed=args.seed,
                    manifest_sha256=file_digest(args.manifest), freeze_saim=args.freeze_saim,
                    resolution=args.resolution)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", local_files_only=True)
    unet.requires_grad_(False)
    unet.to(device=device, dtype=dtype)
    if args.anchor:
        anchor = read_bank(args.anchor)
        if anchor["metadata"].get("role") != "anchor" or anchor["metadata"]["base_model"] != base_model:
            raise ValueError("Expected a normal anchor trained with the same base model")
        if anchor["metadata"].get("resolution") != args.resolution:
            raise ValueError("Anchor and experts must use the same training resolution")
        processors = processors_from_bank(anchor, unet)
        # Optimizer master weights stay fp32 under autocast.
        for processor in processors.values():
            processor.float()
        metadata["anchor_sha256"] = file_digest(args.anchor)
        metadata.update({k: anchor["metadata"][k] for k in ("prompt_blend", "prompt_fg")})
    else:
        processors = create_processors(unet, args.rank)
    modules = nn.ModuleList(list(processors.values())).to(device=device, dtype=torch.float32)
    if args.freeze_saim:
        for processor in processors.values():
            for name, parameter in processor.named_parameters():
                if name.startswith("temporal_"):
                    parameter.requires_grad_(False)
    unet.set_attn_processor(dict(processors))
    unet.train()
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", local_files_only=True)
    vae.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", local_files_only=True)
    encoder.requires_grad_(False).eval().to(device=device, dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", local_files_only=True)
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler", local_files_only=True)
    with torch.no_grad():
        ids = tokenizer([metadata["prompt_blend"], metadata["prompt_fg"]], padding="max_length",
                        max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
        embeddings = encoder(ids.to(device))[0]
    del encoder
    torch.cuda.empty_cache()
    parameters = [p for p in modules.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    start = 0
    if args.resume:
        # Trusted local training checkpoint; unlike bank files this contains RNG tuples.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["metadata"] != metadata:
            raise ValueError("Resume metadata differs from this run")
        if state["run_args"] != {k: v for k, v in vars(args).items() if k not in ("resume", "steps", "output")}:
            raise ValueError("Resume requires the original training options (steps/output may change)")
        modules.load_state_dict(state["modules"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start = state["step"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    # Sampling with replacement avoids hidden sampler position on checkpoint resume.
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    for step in range(start, args.steps):
        loss_sum = 0.0
        for _ in range(args.accumulation):
            batch = [dataset[random.randrange(len(dataset))] for _ in range(args.batch_size)]
            # B,2,C,H,W -> (B*2),C,H,W, matching the original frames=2 processor.
            pixels = torch.stack([torch.stack([item["blend"], item["fg"]]) for item in batch]).flatten(0, 1).to(device)
            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                latents = latents.to(dtype)
                noise = torch.randn_like(latents)
                times = torch.randint(scheduler.config.num_train_timesteps, (args.batch_size,), device=device)
                times = times.repeat_interleave(2)
                noisy = scheduler.add_noise(latents, noise, times)
                target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, times)
            noisy.requires_grad_(args.gradient_checkpointing)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
                prediction = unet(noisy, times, encoder_hidden_states=embeddings.repeat(args.batch_size, 1, 1)).sample
                loss = F.mse_loss(prediction.float(), target.float())
            loss_sum += loss.detach().item() / args.accumulation
            scaler.scale(loss / args.accumulation).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(f"step={step + 1}/{args.steps} loss={loss_sum:.6f}", flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            checkpoint = output / f"checkpoint-{step + 1}"
            save_bank(checkpoint / "adapter_bank.pt", processors, metadata)
            state = dict(step=step + 1, metadata=metadata, modules=modules.state_dict(),
                         run_args={k: v for k, v in vars(args).items() if k not in ("resume", "steps", "output")},
                         optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            torch.save(state, checkpoint / "training_state.pt")
            save_bank(output / "adapter_bank.pt", processors, metadata)


if __name__ == "__main__":
    main(parse_args())
