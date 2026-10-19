"""Per-class JSONL records with synchronized image/mask transforms."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def read_records(manifest, category, normal=False):
    path = Path(manifest).resolve()
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("semantic_status", "confirmed") != "confirmed":
                raise ValueError(f"{path}:{line_number}: class semantics are unconfirmed; "
                                 "review source class meanings before expert/normal training")
            if row.get("class") != category:
                raise ValueError(f"{path}:{line_number}: expected class {category}")
            if not normal and row.get("labels") != [category]:
                raise ValueError(f"{path}:{line_number}: expert training requires labels=[{category!r}]; "
                                 "exclude mixed-defect images in this first version")
            if normal and row.get("labels") != []:
                raise ValueError("Normal training records require labels=[]")
            row["image"] = path.parent / row["image"]
            if not row["image"].is_file():
                raise FileNotFoundError(row["image"])
            if not normal:
                row["mask"] = path.parent / row["mask"]
                if not row["mask"].is_file():
                    raise FileNotFoundError(row["mask"])
            records.append(row)
    if not records:
        raise ValueError(f"Empty manifest: {manifest}")
    return records


def load_pair(row, normal=False):
    with Image.open(row["image"]) as source:
        # Masks must describe the image after EXIF orientation is applied.
        image = ImageOps.exif_transpose(source).convert("RGB")
    if normal:
        mask = Image.new("L", image.size, 0)
    else:
        with Image.open(row["mask"]) as source:
            raw = np.asarray(source)
            if raw.ndim != 2 or not set(np.unique(raw)).issubset({0, 255}):
                raise ValueError(f"Mask must be single-channel binary 0/255: {row['mask']}")
            mask = source.convert("L")
        if mask.size != image.size:
            raise ValueError(f"Image/mask size mismatch: {row['image']}")
        if not np.asarray(mask).any():
            raise ValueError(f"Empty defect mask: {row['mask']}")
    return image, mask


class ManifestDataset(Dataset):
    def __init__(self, manifest, category, size=512, normal=False, augment=True):
        if size <= 0 or size % 8:
            raise ValueError("resolution must be a positive multiple of 8")
        self.records = read_records(manifest, category, normal)
        self.size, self.normal, self.augment = size, normal, augment
        # Fail before loading model weights if any annotation is invalid.
        for row in self.records:
            load_pair(row, normal)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = load_pair(self.records[index], self.normal)
        # Keep the full core and its aspect ratio; identical letterboxing for both.
        scale = self.size / max(image.size)
        shape = tuple(max(1, round(value * scale)) for value in image.size)
        image = image.resize(shape, Image.Resampling.BILINEAR)
        mask = mask.resize(shape, Image.Resampling.NEAREST)
        offset = ((self.size - shape[0]) // 2, (self.size - shape[1]) // 2)
        canvas = Image.new("RGB", (self.size, self.size), (127, 127, 127))
        canvas.paste(image, offset)
        mask_canvas = Image.new("L", canvas.size, 0)
        mask_canvas.paste(mask, offset)
        if self.augment and random.random() < 0.5:
            canvas, mask_canvas = ImageOps.mirror(canvas), ImageOps.mirror(mask_canvas)
        pixels = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 127.5 - 1
        mask_tensor = torch.from_numpy(np.array(mask_canvas)).unsqueeze(0).float() / 255
        # Same convention as baseline: normalized zero (gray) outside anomaly.
        return {"blend": pixels, "fg": pixels * mask_tensor, "mask": mask_tensor}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate a single-class training manifest")
    parser.add_argument("manifest")
    parser.add_argument("--category", required=True)
    parser.add_argument("--normal", action="store_true")
    args = parser.parse_args()
    dataset = ManifestDataset(args.manifest, args.category, normal=args.normal)
    print(f"Validated {len(dataset)} records for {args.category}")
