import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_yolo_dataset import convert, parse_label, rasterize
from manifest_dataset import ManifestDataset


class YoloConversionTests(unittest.TestCase):
    def test_polygon_native_pixels_and_cross_class_overlap(self):
        instances = [dict(class_id=i, geometry="polygon", points=[(.2, .2), (.8, .2), (.8, .8), (.2, .8)]) for i in (0, 1)]
        masks = rasterize(instances, (10, 20))
        self.assertEqual(masks[0].size, (10, 20))
        a, b = np.array(masks[0]), np.array(masks[1])
        self.assertEqual(a[4, 2], 255)
        self.assertEqual(a[0, 0], 0)
        self.assertEqual(a[16, 8], 255)
        self.assertEqual(set(np.unique(a)), {0, 255})
        self.assertTrue(((a > 0) & (b > 0)).any())

    def test_invalid_coordinates_and_box_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.txt"
            path.write_text("0 0.5 0.5 0.2 0.4\n")
            rows = parse_label(path, {0: "a"})
            self.assertEqual(rows[0]["geometry"], "box")
            path.write_text("0 0 0 1.1 0 1 1\n")
            with self.assertRaises(ValueError):
                parse_label(path, {0: "a"})
            path.write_text("2 0 0 1 0 1 1\n")
            with self.assertRaises(ValueError):
                parse_label(path, {0: "a"})

    def test_roundtrip_splits_references_and_unknown_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "converted"
            source.mkdir()
            (source / "data.yaml").write_text("nc: 2\nnames: ['a', 'core']\ntrain: ../train/images\nval: ../valid/images\n")
            for split in ("train", "valid"):
                (source / split / "images").mkdir(parents=True)
                (source / split / "labels").mkdir()
                Image.new("RGB", (32, 24), "gray").save(source / split / "images" / "sample.png")
                (source / split / "labels" / "sample.txt").write_text("0 .1 .1 .8 .1 .8 .8 .1 .8\n")
            report = convert(source, output, previews_per_split=0)
            self.assertEqual(report["splits"]["train"]["images"], 1)
            self.assertEqual(len(report["cross_split_exact_duplicates"]), 1)
            self.assertEqual(report["cross_split_filename_groups"], {"sample": ["train", "valid"]})
            row = json.loads((output / "train/all_samples.jsonl").read_text())
            self.assertTrue((output / "train" / row["image"]).is_file())
            self.assertTrue((output / "train" / row["masks"]["a"]).is_file())
            self.assertFalse(report["training_ready"])
            manifest = output / "train/unconfirmed_single_class/a.jsonl"
            review_row = json.loads(manifest.read_text())
            self.assertTrue((manifest.parent / review_row["image"]).is_file())
            self.assertTrue((manifest.parent / review_row["mask"]).is_file())
            with self.assertRaisesRegex(ValueError, "semantics are unconfirmed"):
                ManifestDataset(manifest, "a")
            with self.assertRaises(FileExistsError):
                convert(source, output)

    def test_missing_annotation_is_not_a_normal_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Missing label"):
                parse_label(Path(tmp) / "missing.txt", {0: "a"})


if __name__ == "__main__":
    unittest.main()
