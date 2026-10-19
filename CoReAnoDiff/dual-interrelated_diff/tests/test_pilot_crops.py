import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_generation_pilot import choose_core, patch_bounds, prepare


class PilotCropTests(unittest.TestCase):
    def test_match_rejects_unrelated_core(self):
        core = [(10, 10), (90, 10), (90, 35), (10, 35)]
        self.assertEqual(choose_core([(20, 20), (30, 20), (30, 25)], [core]), 0)
        self.assertIsNone(choose_core([(20, 70), (30, 70), (30, 80)], [core]))

    def test_crops_preserve_both_disease_masks_and_source_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, yolo = root / "source", root / "yolo"
            for path in [source / "normal", source / "abnormal", yolo / "train/images", yolo / "train/labels"]:
                path.mkdir(parents=True)
            for name, category in [("bad", "abnormal"), ("good", "normal")]:
                image = source / category / (name + ".png")
                Image.new("RGB", (100, 100), "gray").save(image)
                shutil.copyfile(image, yolo / "train/images" / image.name)
                (yolo / "train/labels" / (name + ".txt")).write_text("2 .1 .1 .9 .1 .9 .35 .1 .35\n")
            shapes = [dict(label=c, shape_type="polygon", points=[[x, 20], [x + 10, 20], [x + 10, 25], [x, 25]])
                      for c, x in [("honeycomb", 20), ("groove", 40)]]
            (source / "abnormal/bad.json").write_text(json.dumps(dict(imagePath="bad.png", shapes=shapes)))
            result = prepare(source, yolo, root / "out", normal_limit=1)
            self.assertEqual(result["counts"]["mixed"], 1)
            self.assertEqual(result["counts"]["normal"], 1)
            self.assertEqual(result["counts"]["honeycomb"], 0)
            row = json.loads((root / "out/mixed.jsonl").read_text())
            self.assertEqual(set(row["masks"]), {"honeycomb", "groove"})
            self.assertEqual(row["source_group"], "bad")
            with Image.open(root / "out" / row["image"]) as image:
                for path in row["masks"].values():
                    with Image.open(root / "out" / path) as mask:
                        self.assertEqual(image.size, mask.size)
                        self.assertIsNotNone(mask.getbbox())


if __name__ == "__main__":
    unittest.main()
