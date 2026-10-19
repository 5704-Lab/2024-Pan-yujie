"""Isolate initialization damage from training and regional composition."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from inference_regional import sample
from regional_adapters import create_processors
from regional_runtime import check_local_diffusers


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check_local_diffusers()
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float16,
                                                  safety_checker=None, feature_extractor=None,
                                                  local_files_only=True).to("cuda")
    pipe.vae.float()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    metadata = dict(prompt_blend="a close-up photograph of a cylindrical concrete core sample, realistic concrete texture",
                    prompt_fg="a concrete surface defect on a plain gray background")
    originals = dict(pipe.unet.attn_processors)
    generator = torch.Generator(device="cuda").manual_seed(123)
    x = torch.randn((2, 4, 64, 64), generator=generator, device="cuda", dtype=torch.float16)
    context = pipe._encode_prompt([metadata["prompt_blend"], metadata["prompt_fg"]], "cuda", 1, False)
    def epsilon():
        return pipe.unet(x, 500, encoder_hidden_states=context).sample.float()
    baseline = epsilon()
    results = {}
    masks = torch.ones(1, 512, 512)
    images = []
    for name in ("base", "random_saim", "zero_saim", "zero_saim_legacy_prompt"):
        if name == "random_saim":
            torch.manual_seed(32)
            processors = create_processors(pipe.unet, rank=4, saim_init="legacy_random")
            for processor in processors.values():
                processor.to(device="cuda", dtype=torch.float16)
            pipe.unet.set_attn_processor(dict(processors))
        if name == "zero_saim":
            for processor in processors.values():
                processor.temporal_o.weight.zero_()
                processor.temporal_o.bias.zero_()
        pipe.unet.eval()
        prediction = epsilon()
        results[name] = dict(epsilon_relative_l2=((prediction - baseline).norm() / baseline.norm()).item(),
                             epsilon_max_error=(prediction - baseline).abs().max().item())
        prompts = metadata if name != "zero_saim_legacy_prompt" else dict(prompt_blend="a vfx with sks", prompt_fg="sks")
        image, foreground = sample(pipe, prompts, masks, 512, 512, seed=32, steps=30, guidance=5.5)
        image.save(output / (name + ".png"))
        foreground.save(output / (name + "_foreground.png"))
        images.append((name, image))
    canvas = Image.new("RGB", (512 * len(images), 550), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, image) in enumerate(images):
        draw.text((i * 512 + 6, 5), name + " (UNTRAINED control)", fill="black")
        canvas.paste(image, (i * 512, 35))
    canvas.save(output / "initialization_comparison.jpg")
    (output / "diagnostics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    pipe.unet.set_attn_processor(originals)


if __name__ == "__main__":
    main()
