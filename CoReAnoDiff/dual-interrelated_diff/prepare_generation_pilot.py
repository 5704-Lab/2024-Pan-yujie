"""Create traceable local core patches for an initial generation smoke run.

Use existing YOLO class-2 core outlines only for geometry; disease identities
always come from the new LabelMe annotations. No train/test evaluation split is
inferred. All annotated defects intersecting a crop remain in its label set.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageChops, ImageDraw


def box(points):
    return (math.floor(min(x for x, y in points)), math.floor(min(y for x, y in points)),
            math.ceil(max(x for x, y in points)) + 1, math.ceil(max(y for x, y in points)) + 1)


def local_mask(points, bounds):
    x0, y0, x1, y1 = bounds
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in points], fill=255)
    return mask


def area(mask):
    return mask.histogram()[255]


def choose_core(defect_points, cores):
    bounds = box(defect_points)
    defect = local_mask(defect_points, bounds)
    total = max(1, area(defect))
    scores = [(area(ImageChops.multiply(defect, local_mask(points, bounds))) / total, i)
              for i, points in enumerate(cores)]
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 0.5:
        return None
    return scores[0][1]


def patch_bounds(core, target_points, size):
    x0, y0, x1, y1 = box(core)
    if target_points:
        a, b, c, d = box(target_points)
        if x1 - x0 >= y1 - y0:
            padding = max(24, round((y1 - y0) * .4))
            x0, x1 = max(x0, a - padding), min(x1, c + padding)
        else:
            padding = max(24, round((x1 - x0) * .4))
            y0, y1 = max(y0, b - padding), min(y1, d + padding)
        x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
    else:
        # Normal local patch from the central part of one long core.
        if x1 - x0 > 4 * (y1 - y0):
            mid, half = (x0 + x1) // 2, 2 * (y1 - y0)
            x0, x1 = mid - half, mid + half
        elif y1 - y0 > 4 * (x1 - x0):
            mid, half = (y0 + y1) // 2, 2 * (x1 - x0)
            y0, y1 = mid - half, mid + half
    padding = max(8, round(min(x1 - x0, y1 - y0) * .12))
    return max(0, x0 - padding), max(0, y0 - padding), min(size[0], x1 + padding), min(size[1], y1 + padding)


def prepare(source, yolo, output, normal_limit=32, seed=32):
    source, yolo, output = Path(source).resolve(), Path(yolo).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    lookup = {}
    for split in ("train", "valid", "test"):
        for path in (yolo / split / "labels").glob("*.txt"):
            if path.stem in lookup:
                raise ValueError(f"Duplicate source stem across YOLO splits: {path.stem}")
            lookup[path.stem] = path
    records, rejected = [], []

    def cores_for(image_path):
        label = lookup.get(image_path.stem)
        if label is None:
            raise ValueError("No matching original core outlines")
        original = label.parent.parent / "images" / image_path.name
        if hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(image_path.read_bytes()).digest():
            raise ValueError("Image differs from original core-label source")
        with Image.open(image_path) as im:
            width, height = im.size
        cores = []
        for line in label.read_text().splitlines():
            values = list(map(float, line.split()))
            if values and values[0] == 2:
                if len(values) < 7 or len(values) % 2 != 1:
                    raise ValueError("Expected a class-2 core polygon")
                cores.append([(x * width, y * height) for x, y in zip(values[1::2], values[2::2])])
        return cores, label

    def write_crop(image_path, image, bounds, shapes, index, label_path):
        key = f"{image_path.stem}_core_{index:03d}"
        pixels = image.crop(bounds)
        image_name = f"images/{key}.png"
        pixels.save(output / image_name)
        masks = {}
        for shape in shapes:
            mask = local_mask(shape["points"], bounds)
            if area(mask):
                category = shape["label"]
                masks[category] = ImageChops.lighter(masks[category], mask) if category in masks else mask
        paths = {}
        for category, mask in masks.items():
            paths[category] = f"masks/{key}_{category}.png"
            mask.save(output / paths[category])
        records.append(dict(image=image_name, masks=paths, labels=sorted(paths),
                            source_image=str(image_path), source_core_label=str(label_path),
                            source_group=image_path.stem.split(".rf.")[0], crop_xyxy=list(bounds),
                            core_instance_index=index, split="pilot_training_only", specimen_id=None))

    for annotation in sorted((source / "abnormal").glob("*.json")):
        data = json.loads(annotation.read_text())
        image_path = annotation.parent / data["imagePath"]
        try:
            cores, label_path = cores_for(image_path)
        except (ValueError, FileNotFoundError) as error:
            rejected.append(dict(source=str(annotation), reason=str(error)))
            continue
        groups = {}
        for index, shape in enumerate(data["shapes"]):
            if shape.get("shape_type", "polygon") != "polygon":
                raise ValueError(f"Only disease polygons supported: {annotation}")
            match = choose_core(shape["points"], cores)
            if match is None:
                rejected.append(dict(source=str(annotation), shape=index, reason="Less than 50% defect/core overlap"))
            else:
                groups.setdefault(match, []).extend(shape["points"])
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            for core_index, points in groups.items():
                bounds = patch_bounds(cores[core_index], points, image.size)
                # Preserve all annotated categories intersecting the patch,
                # including nearby defects on other cores in the same crop.
                write_crop(image_path, image, bounds, data["shapes"], core_index, label_path)
    normals = sorted(p for p in (source / "normal").glob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    random.Random(seed).shuffle(normals)
    normal_count = 0
    for image_path in normals:
        if normal_count >= normal_limit:
            break
        try:
            cores, label_path = cores_for(image_path)
            if not cores:
                raise ValueError("No core instances")
        except (ValueError, FileNotFoundError) as error:
            rejected.append(dict(source=str(image_path), reason=str(error)))
            continue
        index = max(range(len(cores)), key=lambda i: (box(cores[i])[2] - box(cores[i])[0]) * (box(cores[i])[3] - box(cores[i])[1]))
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            write_crop(image_path, image, patch_bounds(cores[index], [], image.size), [], index, label_path)
        normal_count += 1
    manifests = {"all_samples": records, "mixed": [r for r in records if len(r["labels"]) > 1]}
    for category in ("normal", "honeycomb", "groove", "mud"):
        rows = []
        for row in records:
            if row["labels"] == ([] if category == "normal" else [category]):
                item = dict(row, **{"class": category})
                if category != "normal":
                    item["mask"] = row["masks"][category]
                rows.append(item)
        manifests[category] = rows
    for name, rows in manifests.items():
        if rows:
            (output / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = dict(counts={k: len(v) for k, v in manifests.items()}, rejected=rejected,
                  purpose="pilot_training_only_not_evaluation", normal_source="user_normal_folder",
                  note="Local core patches preserve all intersecting disease labels; source group recorded. Manual crop review pending.")
    (output / "preparation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("yolo")
    p.add_argument("output")
    p.add_argument("--normal-limit", type=int, default=32)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.yolo, a.output, a.normal_limit), indent=2))
