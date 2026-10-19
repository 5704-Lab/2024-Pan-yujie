"""Layout generation and reference-image editing using routed dual adapters."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from regional_adapters import file_digest, install_regional_adapters, read_bank
from regional_runtime import check_local_diffusers


def load_layout(regions, size, root, feather=0):
    masks = []
    for region in regions:
        with Image.open(root / region["mask"]) as source:
            if source.size != size or source.mode != "L":
                raise ValueError(f"Layout must be single-channel L with size {size}: {region['mask']}")
            mask = source.copy()
        if feather:
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        masks.append(torch.from_numpy(np.array(mask)).float() / 255)
    return torch.stack(masks)


@torch.no_grad()
def sample(pipe, metadata, masks, height, width, seed=32, steps=50, guidance=2.5,
           reference=None, strength=0.75):
    """DDIM, one interleaved whole/foreground pair, with optional masked editing.

    Editing anchors unrequested whole-image latents at every noise level. A final
    image composite is exported separately; that pixel preservation is not a
    property of the regional adapter itself.
    """
    device, dtype = pipe.unet.device, pipe.unet.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    cfg = guidance > 1
    blend = pipe._encode_prompt(metadata["prompt_blend"], device, 1, cfg)
    fg = pipe._encode_prompt(metadata["prompt_fg"], device, 1, cfg)
    embeddings = torch.stack([blend, fg], dim=1).flatten(0, 1)
    pipe.scheduler.set_timesteps(steps, device=device)
    times = pipe.scheduler.timesteps
    shape = (2, pipe.unet.config.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    union = masks.amax(dim=0)[None, None].to(device=device, dtype=dtype)
    latent_mask = F.interpolate(union, size=shape[-2:], mode="bilinear", align_corners=False)
    original = None
    if reference is not None:
        pixels = torch.from_numpy(np.array(reference)).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32) / 127.5 - 1
        pair = torch.cat([pixels, pixels * union.float()])
        original = pipe.vae.encode(pair).latent_dist.mode().to(dtype) * pipe.vae.config.scaling_factor
        count = max(1, int(steps * strength))
        times = times[steps - count:]
        latents = pipe.scheduler.add_noise(original, noise, times[0].reshape(1))
    else:
        latents = noise * pipe.scheduler.init_noise_sigma
    for index, timestep in enumerate(pipe.progress_bar(times)):
        inputs = torch.cat([latents, latents]) if cfg else latents
        inputs = pipe.scheduler.scale_model_input(inputs, timestep)
        prediction = pipe.unet(inputs, timestep, encoder_hidden_states=embeddings).sample
        if cfg:
            uncond, cond = prediction.chunk(2)
            prediction = uncond + guidance * (cond - uncond)
        latents = pipe.scheduler.step(prediction, timestep, latents, generator=generator).prev_sample
        if original is not None:
            reference_at_next = (
                pipe.scheduler.add_noise(original, noise, times[index + 1].reshape(1))
                if index + 1 < len(times) else original
            )
            # Only the whole-image branch is anchored; the foreground branch
            # remains an anomaly generator and participates in SAIM.
            latents[:1] = latent_mask * latents[:1] + (1 - latent_mask) * reference_at_next[:1]
    decoded = pipe.vae.decode(latents.float() / pipe.vae.config.scaling_factor).sample
    array = ((decoded / 2 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(item) for item in array]


def main(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mode = config.get("mode", "generate")
    if mode not in ("generate", "edit"):
        raise ValueError("mode must be generate or edit")
    height, width = config.get("height", 512), config.get("width", 512)
    if min(height, width) < 8 or height % 8 or width % 8:
        raise ValueError("height and width must be positive multiples of 8")
    regions = config["regions"]
    if not regions:
        raise ValueError("At least one region is required")
    feather = float(config.get("feather", 0))
    if not math.isfinite(feather) or feather < 0:
        raise ValueError("feather must be finite and nonnegative")
    masks = load_layout(regions, (width, height), root, feather)
    steps = int(config.get("steps", 50))
    strength = float(config.get("edit_strength", 0.75))
    guidance = float(config.get("guidance", 2.5))
    if steps < 1 or not 0 < strength <= 1 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Invalid steps, edit_strength or guidance")
    base_model = str((root / config["base_model"]).resolve())
    anchor_path = root / config["anchor"]
    anchor = read_bank(anchor_path)
    if anchor["metadata"]["base_model"] != base_model:
        raise ValueError("Base model path must match training metadata")
    for region in regions:
        expert = read_bank(root / region["adapter"])
        if expert["metadata"].get("category") != region["class"]:
            raise ValueError(f"Expert class mismatch: {region['class']}")
    reference = None
    if mode == "edit":
        with Image.open(root / config["reference"]) as source:
            reference = ImageOps.pad(ImageOps.exif_transpose(source).convert("RGB"), (width, height),
                                     method=Image.Resampling.BILINEAR, color=(127, 127, 127))
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    seeds = config.get("seeds", [32])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of unique integers")
    for seed in seeds:
        if not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if (output / f"seed-{seed}").exists():
            raise FileExistsError(f"Output already exists: {output / f'seed-{seed}'}")
    check_local_diffusers()
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for full SD1.5 inference")
    dtype = torch.float16 if config.get("precision", "fp16") == "fp16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype,
                                                   safety_checker=None, feature_extractor=None,
                                                   local_files_only=True).to("cuda")
    pipe.vae.to(dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    handle = install_regional_adapters(
        pipe.unet, anchor_path, [root / r["adapter"] for r in regions], masks,
        [r.get("strength", 1.0) for r in regions], config.get("interpolation", "bilinear"),
        config.get("shared_saim", False),
    )
    pipe.unet.eval()
    try:
        for seed in seeds:
            image, foreground = sample(pipe, anchor["metadata"], masks, height, width, seed,
                                       steps, guidance, reference, strength)
            target = output / f"seed-{seed}"
            target.mkdir()
            image.save(target / "image_raw.png")
            foreground.save(target / "foreground.png")
            if reference is not None:
                alpha = Image.fromarray((masks.amax(dim=0).numpy() * 255).round().astype(np.uint8))
                Image.composite(image, reference, alpha).save(target / "image.png")
                reference.save(target / "reference_aligned.png")
            else:
                image.save(target / "image.png")
            class_masks = {}
            for index, (region, mask) in enumerate(zip(regions, masks)):
                Image.fromarray((mask.numpy() * 255).round().astype(np.uint8)).save(target / f"requested_region_{index}.png")
                category = region["class"]
                class_masks[category] = torch.maximum(class_masks.get(category, torch.zeros_like(mask)), mask)
            label_files = {}
            for index, (category, mask) in enumerate(class_masks.items()):
                filename = f"requested_class_{index}.png"
                Image.fromarray((mask.numpy() * 255).round().astype(np.uint8)).save(target / filename)
                label_files[category] = filename
            metadata = dict(config=config, seed=seed, mode=mode, requested_class_masks=label_files,
                            mask_status="requested_layout_not_verified_ground_truth",
                            anchor_sha256=file_digest(anchor_path),
                            expert_sha256=[file_digest(root / r["adapter"]) for r in regions],
                            pixel_composite=reference is not None)
            (target / "generation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    main(parser.parse_args().config)
