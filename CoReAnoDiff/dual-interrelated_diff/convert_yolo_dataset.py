"""YOLO annotation conversion with explicit, unconfirmed class semantics.

Preserve source images and split membership. Rasterize polygons at native image
resolution; boxes, if present, are recorded as coarse box regions, not segmentation.
No disease identity or normal status is inferred from opaque class names.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [(240, 65, 65), (55, 125, 255), (55, 200, 90), (240, 190, 30)]


def parse_label(path, names):
    if not path.is_file():
        raise ValueError(f"Missing label; do not assume normal: {path}")
    instances = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(token) for token in line.split()]
        class_id = int(values[0])
        if values[0] != class_id or class_id not in names:
            raise ValueError(f"{path}:{number}: invalid class ID")
        coordinates = values[1:]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in coordinates):
            raise ValueError(f"{path}:{number}: nonfinite or out-of-range coordinates")
        if len(coordinates) == 4:
            x, y, width, height = coordinates
            if min(width, height) <= 0:
                raise ValueError(f"{path}:{number}: empty box")
            geometry = "box"
            points = [(max(0, x - width / 2), max(0, y - height / 2)),
                      (min(1, x + width / 2), min(1, y + height / 2))]
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            geometry = "polygon"
            points = list(zip(coordinates[::2], coordinates[1::2]))
            if len(set(points)) < 3:
                raise ValueError(f"{path}:{number}: degenerate polygon")
        else:
            raise ValueError(f"{path}:{number}: invalid YOLO box/polygon record")
        instances.append(dict(class_id=class_id, source_line=number, geometry=geometry, points=points))
    return instances


def rasterize(instances, size):
    """Merge same-class instances; preserve independent cross-class overlap."""
    width, height = size
    masks = {}
    for instance in instances:
        mask = masks.setdefault(instance["class_id"], Image.new("L", size, 0))
        # YOLO stores x/width and y/height; clip the endpoint at the image edge.
        points = [(min(width - 1, max(0, round(x * width))),
                   min(height - 1, max(0, round(y * height)))) for x, y in instance["points"]]
        if instance["geometry"] == "box":
            ImageDraw.Draw(mask).rectangle(points, fill=255)
        else:
            ImageDraw.Draw(mask).polygon(points, fill=255)
    if any(mask.getbbox() is None for mask in masks.values()):
        raise ValueError("Rasterization produced an empty class mask")
    return masks


def preview(image_path, masks, names, target,
            title="Original (left) | YOLO class masks (right); disease meanings UNCONFIRMED"):
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    original.thumbnail((740, 520))
    overlay = original.convert("RGBA")
    for class_id, mask in sorted(masks.items()):
        small_mask = mask.resize(original.size, Image.Resampling.NEAREST)
        alpha = small_mask.point(lambda value: 90 if value else 0)
        layer = Image.new("RGBA", original.size, COLORS[class_id % len(COLORS)] + (0,))
        layer.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, layer)
    canvas = Image.new("RGB", (original.width * 2, original.height + 45), "white")
    canvas.paste(original, (0, 45))
    canvas.paste(overlay.convert("RGB"), (original.width, 45))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 3), title, fill="black")
    x = 6
    for class_id in sorted(masks):
        text = f"{class_id}: {names[class_id]}"
        draw.rectangle((x, 23, x + 12, 35), fill=COLORS[class_id % len(COLORS)])
        draw.text((x + 16, 23), text, fill="black")
        x += 145
    canvas.save(target, quality=90)


def convert(source, output, previews_per_split=4):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8-sig"))
    names = config["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if len(names) != config.get("nc", len(names)) or len(set(names.values())) != len(names):
        raise ValueError("Invalid/duplicate class names")
    if any(not isinstance(name, str) or not name.replace("_", "").isalnum() for name in names.values()):
        raise ValueError("Class names must contain only letters, numbers and underscores")
    # This export's YAML uses ../train/images although the actual folders are
    # beside data.yaml. Prefer those explicit local folders without editing YAML.
    splits = {}
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        local = source / split / "images"
        configured = config.get(key)
        if not local.is_dir() and configured:
            local = source / configured
        if not local.is_dir():
            continue
        images = sorted(p for p in local.iterdir() if p.suffix.lower() in EXTENSIONS)
        if not images:
            continue
        label_root = local.parent / "labels"
        if len({image.stem for image in images}) != len(images):
            raise ValueError(f"Duplicate image stems in {split}")
        extras = {p.stem for p in label_root.glob("*.txt")} - {p.stem for p in images}
        if extras:
            raise ValueError(f"Unpaired labels in {split}: {sorted(extras)[:5]}")
        records = []
        for image in images:
            label = label_root / (image.stem + ".txt")
            instances = parse_label(label, names)
            with Image.open(image) as im:
                if im.getexif().get(274, 1) != 1:
                    raise ValueError(f"EXIF orientation must be normalized before conversion: {image}")
                size = im.size
            with Image.open(image) as im:
                im.verify()
            records.append((image, label, size, instances))
        splits[split] = records
    if not splits:
        raise ValueError("No nonempty YOLO splits found")
    output.mkdir(parents=True)
    report = dict(source=str(source), class_names=names, semantic_status="unconfirmed",
                  normal_images_confirmed=0, training_ready=False, splits={},
                  missing_splits=[s for s in ("train", "valid", "test") if s not in splits],
                  notes=["Source images are referenced, not copied or modified.",
                         "Class names do not establish disease identities or normal status.",
                         "No automatic disease expert manifests have been created.",
                         "No inferred specimen identity: split leakage between related photographs remains unverified."])
    (output / "INCOMPLETE").write_text("Conversion in progress; do not consume manifests yet.\n")
    digests, groups = defaultdict(list), defaultdict(set)
    for split, records in splits.items():
        split_root = output / split
        (split_root / "masks").mkdir(parents=True)
        (split_root / "previews").mkdir()
        rows, single_rows = [], defaultdict(list)
        class_images, class_instances, geometry, combinations = Counter(), Counter(), Counter(), Counter()
        preview_count, preview_classes = 0, set()
        for index, (image, label, size, instances) in enumerate(records):
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            digests[digest].append(dict(split=split, image=image.name))
            group = image.stem.split(".rf.")[0]
            groups[group].add(split)
            masks = rasterize(instances, size)
            mask_paths = {}
            for class_id, mask in sorted(masks.items()):
                name = names[class_id]
                folder = split_root / "masks" / name
                folder.mkdir(exist_ok=True)
                mask_path = folder / (image.stem + ".png")
                mask.save(mask_path)
                mask_paths[name] = str(mask_path.relative_to(split_root))
                class_images[name] += 1
            for instance in instances:
                class_instances[names[instance["class_id"]]] += 1
                geometry[instance["geometry"]] += 1
            labels = [names[i] for i in sorted(masks)]
            combinations[" + ".join(labels) or "unlabeled"] += 1
            row = dict(image=os.path.relpath(image, split_root), masks=mask_paths, labels=labels,
                       width=size[0], height=size[1], split=split, semantic_status="unconfirmed",
                       annotation_quality="source_polygon" if all(i["geometry"] == "polygon" for i in instances) and instances
                       else "coarse_box_region" if instances else "unlabeled",
                       source_label=os.path.relpath(label, split_root), source_image_sha256=digest,
                       source_group=group, specimen_id=None)
            rows.append(row)
            if len(labels) == 1:
                single_rows[labels[0]].append(dict(row, mask=mask_paths[labels[0]], **{"class": labels[0]}))
            if preview_count < previews_per_split and (not set(masks).issubset(preview_classes) or index < previews_per_split):
                preview(image, masks, names, split_root / "previews" / (image.stem + ".jpg"))
                preview_count += 1
                preview_classes.update(masks)
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                print(f"{split}: {index + 1}/{len(records)}", flush=True)
        (split_root / "all_samples.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        if single_rows:
            folder = split_root / "unconfirmed_single_class"
            folder.mkdir()
            for name, values in single_rows.items():
                # Review manifests live one directory below the split root.
                for row in values:
                    row["image"] = "../" + row["image"]
                    row["mask"] = "../" + row["mask"]
                    row["masks"] = {k: "../" + v for k, v in row["masks"].items()}
                    row["source_label"] = "../" + row["source_label"]
                (folder / (name + ".jsonl")).write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
        report["splits"][split] = dict(images=len(records), class_images=dict(class_images),
                                      class_instances=dict(class_instances), geometry=dict(geometry),
                                      label_combinations=dict(combinations), unlabeled_images=sum(not r[3] for r in records),
                                      single_raw_class_images=sum(len(v) for v in single_rows.values()),
                                      multi_raw_class_images=sum(len(row["labels"]) > 1 for row in rows))
    report["cross_split_exact_duplicates"] = [items for items in digests.values() if len({x["split"] for x in items}) > 1]
    report["cross_split_filename_groups"] = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "class_mapping.pending.json").write_text(json.dumps(
        {str(i): {"source_name": name, "meaning": None, "role": "unconfirmed"} for i, name in names.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "INCOMPLETE").unlink()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    result = convert(args.source, args.output)
    print(json.dumps(result["splits"], ensure_ascii=False, indent=2))
