"""Export LabelMe polygon/rectangle annotations as multilabel PNGs + JSONL.

Input directory must contain only the already selected training partition.
Empty annotations are not automatically treated as normal specimens.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", help="Directory containing LabelMe JSON files")
    parser.add_argument("output", help="New export directory")
    parser.add_argument("--classes", nargs="+", default=["honeycomb", "groove", "pitting", "mud", "fracture"])
    parser.add_argument("--normal-images", help="User-confirmed normal image directory; no JSON required")
    parser.add_argument("--reference-images", action="store_true", help="Reference source images instead of writing PNG copies")
    args = parser.parse_args()
    if len(set(args.classes)) != len(args.classes) or set(args.classes) & {"normal", "core"}:
        raise ValueError("Class names must be unique, excluding reserved normal/core")
    if any(not name.replace("_", "").isalnum() for name in args.classes):
        raise ValueError("Use alphanumeric/underscore class names")
    sources = sorted(Path(args.annotations).glob("*.json"))
    if not sources:
        raise ValueError("No LabelMe JSON files found")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new export directory to avoid stale manifests or overwritten masks")
    # Validate all annotations before creating output.
    samples = []
    instance_counts = Counter()
    for source in sources:
        annotation = json.loads(source.read_text(encoding="utf-8"))
        image_path = source.parent / annotation["imagePath"]
        with Image.open(image_path) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"Normalize EXIF orientation before annotation: {image_path}")
            image_size = image.size
        size = (annotation["imageWidth"], annotation["imageHeight"])
        if image_size != size:
            raise ValueError(f"Annotation/image size mismatch: {source}")
        masks = {}
        for shape in annotation["shapes"]:
            category = shape["label"]
            if category not in args.classes + ["core"]:
                raise ValueError(f"Unknown label {category!r}: {source}")
            mask = masks.setdefault(category, Image.new("L", size, 0))
            points = [tuple(point) for point in shape["points"]]
            if any(len(point) != 2 or not all(math.isfinite(value) for value in point)
                   or not (0 <= point[0] < size[0] and 0 <= point[1] < size[1]) for point in points):
                raise ValueError(f"Nonfinite or out-of-image annotation points: {source}")
            kind = shape.get("shape_type", "polygon")
            if kind == "polygon" and len(points) >= 3:
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
                if abs(area) < 1e-6:
                    raise ValueError(f"Degenerate polygon: {source}")
                ImageDraw.Draw(mask).polygon(points, fill=255)
            elif kind == "rectangle" and len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                ImageDraw.Draw(mask).rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=255)
            else:
                raise ValueError(f"Only polygon/rectangle shapes supported: {source}")
            instance_counts[category] += 1
        labels = sorted(set(masks) - {"core"})
        if not labels and not annotation.get("flags", {}).get("normal", False):
            raise ValueError(f"Empty annotations require flags.normal=true: {source}")
        if labels and annotation.get("flags", {}).get("normal", False):
            raise ValueError(f"Normal flag conflicts with defect labels: {source}")
        if any(not np.asarray(mask).any() for mask in masks.values()):
            raise ValueError(f"Empty polygon rasterization: {source}")
        samples.append((source, annotation, image_path.resolve(), masks, labels))
    normal_paths = []
    if args.normal_images:
        normal_root = Path(args.normal_images).resolve()
        if not normal_root.is_dir():
            raise FileNotFoundError(normal_root)
        normal_paths = sorted(p for p in normal_root.rglob("*") if p.is_file()
                              and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
        abnormal_paths = {sample[2] for sample in samples}
        if any(path in abnormal_paths for path in normal_paths):
            raise ValueError("An image occurs in both normal and annotated inputs")
        for path in normal_paths:
            with Image.open(path) as image:
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(f"Normalize EXIF orientation before export: {path}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    manifests = {category: [] for category in ["normal"] + args.classes}
    all_rows = []
    for source, annotation, image_path, masks, labels in samples:
        sample_id = source.stem
        if args.reference_images:
            image_name = os.path.relpath(image_path, output)
        else:
            image_name = f"images/{sample_id}.png"
            with Image.open(image_path) as image:
                image.convert("RGB").save(output / image_name)
        mask_paths = {}
        for category, mask in masks.items():
            mask_name = f"masks/{sample_id}_{category}.png"
            mask.save(output / mask_name)
            mask_paths[category] = mask_name
        row = dict(image=image_name, labels=labels, masks=mask_paths,
                   specimen_id=annotation.get("specimen_id"), source_annotation=str(source.resolve()))
        all_rows.append(row)
        if not labels:
            manifests["normal"].append(dict(row, **{"class": "normal"}))
        elif len(labels) == 1:
            category = labels[0]
            manifests[category].append(dict(row, mask=mask_paths[category], **{"class": category}))
    for index, path in enumerate(normal_paths):
        if args.reference_images:
            image_name = os.path.relpath(path, output)
        else:
            image_name = f"images/normal_{index:06d}.png"
            with Image.open(path) as image:
                image.convert("RGB").save(output / image_name)
        row = dict(image=image_name, labels=[], masks={}, specimen_id=None,
                   source_image=str(path), normal_label_source="user_provided_normal_directory")
        all_rows.append(row)
        manifests["normal"].append(dict(row, **{"class": "normal"}))
    mixed_rows = [row for row in all_rows if len(row["labels"]) > 1]
    for category, records in list(manifests.items()) + [("all_samples", all_rows), ("mixed", mixed_rows)]:
        if records:
            (output / f"{category}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    statistics = {"unique_images": len(all_rows),
                  "label_occurrences": {category: sum(category in row["labels"] for row in all_rows)
                                        for category in args.classes},
                  "single_class_training": {category: len(rows) for category, rows in manifests.items()},
                  "mixed_images": len(mixed_rows),
                  "annotation_instances": dict(instance_counts),
                  "unannotated_images": sorted(p.name for p in Path(args.annotations).iterdir()
                                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                                               and p.stem not in {source.stem for source in sources})}
    (output / "statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
