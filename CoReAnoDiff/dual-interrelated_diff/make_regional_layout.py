"""Sample disjoint requested regions inside an explicitly supplied core mask."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core_mask", help="Binary 0/255 mask of editable concrete core pixels")
    parser.add_argument("output")
    parser.add_argument("--classes", nargs="+", default=["honeycomb", "groove"])
    parser.add_argument("--seed", type=int, default=32)
    args = parser.parse_args()
    with Image.open(args.core_mask) as image:
        core = np.array(image)
    if core.ndim != 2 or not set(np.unique(core)).issubset({0, 255}):
        raise ValueError("core_mask must be a single-channel binary 0/255 image")
    rng = np.random.default_rng(args.seed)
    available = core > 0
    height, width = available.shape
    masks = []
    for category in args.classes:
        locations = np.argwhere(available)
        if not len(locations):
            raise ValueError("Not enough core area for requested classes")
        y, x = locations[rng.integers(len(locations))]
        # Shape prior is only a layout heuristic, not an anatomical annotation.
        region_h = max(1, round(height * (0.05 if category == "groove" else 0.15)))
        region_w = max(1, round(width * 0.2))
        region = np.zeros_like(available)
        region[max(0, y - region_h // 2):min(height, y + region_h // 2 + 1),
               max(0, x - region_w // 2):min(width, x + region_w // 2 + 1)] = True
        region &= available
        available &= ~region
        masks.append(region)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (category, mask) in enumerate(zip(args.classes, masks)):
        name = f"region_{index}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / name)
        records.append({"class": category, "mask": name, "strength": 1.0})
    (output / "layout.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
