"""Train 20-step pilot banks and generate three explicitly marked smoke outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data, base, output = Path(args.data).resolve(), Path(args.base_model).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(purpose="smoke_test_only_not_quality_validated", steps_per_adapter=args.steps,
                   normal_status="user_supplied_not_individually_reviewed", outputs=[], status="training")
    summary_path = output / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    for category in ("normal", "honeycomb", "groove"):
        target = output / "adapters" / category
        bank = target / "adapter_bank.pt"
        if bank.exists():
            # Allows restarting generation after fixing a downstream integration
            # issue, without repeating completed smoke training.
            from regional_adapters import read_bank, file_digest
            metadata = read_bank(bank)["metadata"]
            if metadata["category"] != category or metadata["base_model"] != str(base):
                raise ValueError("Existing smoke bank is from another run")
            run = json.loads((target / "run.json").read_text())
            if run["steps"] != args.steps or metadata["manifest_sha256"] != file_digest(data / (category + ".jsonl")):
                raise ValueError("Existing smoke bank used different steps or training records")
            continue
        command = [sys.executable, str(root / "train_regional_lora.py"), "--base-model", str(base),
                   "--manifest", str(data / (category + ".jsonl")), "--category", category,
                   "--role", "anchor" if category == "normal" else "expert", "--output", str(target),
                   "--steps", str(args.steps), "--save-every", str(args.steps), "--accumulation", "1",
                   "--batch-size", "1", "--seed", "32"]
        if category != "normal":
            command += ["--anchor", str(output / "adapters/normal/adapter_bank.pt")]
        subprocess.run(command, cwd=root, check=True)
    summary["status"] = "sampling"
    summary_path.write_text(json.dumps(summary, indent=2))
    layouts = output / "layouts"
    layouts.mkdir(exist_ok=True)
    for category, bounds in (("honeycomb", (100, 205, 235, 300)), ("groove", (280, 225, 410, 275))):
        mask = Image.new("L", (512, 512), 0)
        ImageDraw.Draw(mask).rectangle(bounds, fill=255)
        mask.save(layouts / (category + ".png"))
    for name, categories in (("honeycomb", ["honeycomb"]), ("groove", ["groove"]),
                              ("combined", ["honeycomb", "groove"])):
        destination = output / "images" / name
        config = dict(base_model=str(base), anchor=str(output / "adapters/normal/adapter_bank.pt"),
                      output=str(destination), mode="generate", height=512, width=512, steps=20,
                      guidance=2.5, seeds=[32], precision="fp16", interpolation="bilinear", feather=2.0,
                      regions=[dict(**{"class": c}, adapter=str(output / "adapters" / c / "adapter_bank.pt"),
                                    mask=str(layouts / (c + ".png")), strength=1.0) for c in categories])
        path = output / (name + ".json")
        path.write_text(json.dumps(config, indent=2))
        if not (destination / "seed-32/generation.json").is_file():
            subprocess.run([sys.executable, str(root / "inference_regional.py"), str(path)], cwd=root, check=True)
        summary["outputs"].append(str(destination / "seed-32/image.png"))
        summary_path.write_text(json.dumps(summary, indent=2))
    canvas = Image.new("RGB", (512 * 3, 550), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, path) in enumerate(zip(("honeycomb", "groove", "combined"), summary["outputs"])):
        draw.text((index * 512 + 5, 5), name + f" | {args.steps}-step adapter SMOKE TEST", fill="black")
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"), (index * 512, 35))
    canvas.save(output / "smoke_contact_sheet.jpg")
    summary["status"] = "completed_smoke_test_not_quality_validated"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
