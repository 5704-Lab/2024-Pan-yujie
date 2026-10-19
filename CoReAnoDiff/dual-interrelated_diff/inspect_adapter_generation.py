"""Generate raw single-bank outputs, isolating adapter quality from routing."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from inference_regional import sample
from regional_adapters import file_digest, read_bank, processors_from_bank
from regional_runtime import check_local_diffusers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--banks", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[32, 33])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    check_local_diffusers()
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=torch.float16,
                                                  safety_checker=None, feature_extractor=None,
                                                  local_files_only=True).to("cuda")
    pipe.vae.float()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    panels, records = [], []
    for path in args.banks:
        path = Path(path).resolve()
        bank = read_bank(path)
        metadata = bank["metadata"]
        if metadata["base_model"] != str(Path(args.base_model).resolve()):
            raise ValueError("Base model does not match adapter")
        pipe.unet.set_attn_processor(processors_from_bank(bank, pipe.unet))
        pipe.unet.eval()
        name = metadata["category"]
        for seed in args.seeds:
            image, foreground = sample(pipe, metadata, torch.zeros(1, 512, 512), 512, 512,
                                       seed=seed, steps=50, guidance=5.5)
            image.save(output / f"{name}_seed_{seed}.png")
            foreground.save(output / f"{name}_seed_{seed}_foreground.png")
            panels.append((f"{name} | seed {seed} | raw single adapter", image))
        records.append(dict(bank=str(path), sha256=file_digest(path), metadata=metadata))
    canvas = Image.new("RGB", (512 * len(args.banks), 550 * len(args.seeds)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        column, row = divmod(index, len(args.seeds))
        x, y = column * 512, row * 550
        draw.text((x + 5, y + 5), title, fill="black")
        canvas.paste(image, (x, y + 35))
    canvas.save(output / "single_adapter_comparison.jpg")
    (output / "provenance.json").write_text(json.dumps(dict(records=records, seeds=args.seeds,
                                                          routing=False, reference_image=False,
                                                          quality_status="pending_visual_review"), indent=2))


if __name__ == "__main__":
    main()
