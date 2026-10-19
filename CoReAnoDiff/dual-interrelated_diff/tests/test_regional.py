"""CPU contract tests using the actual vendored dual-branch processor classes.

Extract only the two class definitions through AST to isolate attention tests
from optional pipeline packages (transformers, hub, torchvision, etc.). No
processor implementation is copied or replaced. Full SD integration is separate.
"""
import ast
import json
from pathlib import Path
import random
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manifest_dataset import ManifestDataset
from inference_regional import sample
from regional_adapters import (RegionalAttnProcessor, file_digest, install_regional_adapters,
                               normalized_weights, read_bank, save_bank)


def vendored_processor():
    path = ROOT / "diffusers/models/attention_processor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("LoRALinearLayer", "LoRAAttnProcessor2_0")]
    namespace = dict(torch=torch, nn=nn, F=F, einops=einops, Attention=nn.Module)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["LoRAAttnProcessor2_0"]


PROCESSOR = vendored_processor()


class SmallAttention(nn.Module):
    def __init__(self, context=8):
        super().__init__()
        self.to_q = nn.Linear(8, 8, bias=False)
        self.to_k = nn.Linear(context, 8, bias=False)
        self.to_v = nn.Linear(context, 8, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(8, 8), nn.Dropout(0)])
        self.heads = 2
        self.group_norm, self.norm_cross = None, False
        self.residual_connection, self.rescale_output_factor = True, 1

    def forward(self, hidden_states, encoder_hidden_states=None):
        return self.processor(self, hidden_states, encoder_hidden_states=encoder_hidden_states)


class SmallTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Module()])
        self.transformer_blocks[0].attn1 = SmallAttention()
        self.transformer_blocks[0].attn1.processor = PROCESSOR(8, rank=2)

    def forward(self, pixels):
        return self.transformer_blocks[0].attn1(pixels.flatten(2).transpose(1, 2))


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = SmallTransformer()

    @property
    def attn_processors(self):
        return {"block.transformer_blocks.0.attn1.processor": self.block.transformer_blocks[0].attn1.processor}

    def set_attn_processor(self, processors):
        self.block.transformer_blocks[0].attn1.processor = processors.pop(next(iter(processors)))


class RegionalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.attn = SmallAttention()
        self.anchor = PROCESSOR(8, rank=2)
        self.expert = PROCESSOR(8, rank=2)
        # Include nonzero LoRA residuals in both frames, not only temporal output.
        for processor in (self.anchor, self.expert):
            for name, param in processor.named_parameters():
                if "lora" in name and "up.weight" in name:
                    nn.init.normal_(param, std=0.1)
        self.hidden = torch.randn(4, 6, 8)  # CFG x two branches, rectangular 2x3

    def route(self, masks, strengths=None, experts=None):
        experts = experts or [self.expert]
        router = RegionalAttnProcessor(self.anchor, experts, masks, strengths or [1.] * len(experts))
        router.spatial_shape = (2, 3)
        return router

    def test_zero_layout_returns_anchor(self):
        router = self.route(torch.zeros(1, 2, 3))
        torch.testing.assert_close(router(self.attn, self.hidden), self.anchor(self.attn, self.hidden), rtol=0, atol=0)

    def test_full_layout_returns_complete_expert(self):
        router = self.route(torch.ones(1, 2, 3))
        torch.testing.assert_close(router(self.attn, self.hidden), self.expert(self.attn, self.hidden))

    def test_zero_strength_returns_anchor(self):
        router = RegionalAttnProcessor(self.anchor, [self.expert], torch.ones(1, 2, 3), [0.])
        router.spatial_shape = (2, 3)
        torch.testing.assert_close(router(self.attn, self.hidden), self.anchor(self.attn, self.hidden), rtol=0, atol=0)

    def test_local_output_and_cfg_order(self):
        mask = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        result = self.route(mask)(self.attn, self.hidden)
        anchor = self.anchor(self.attn, self.hidden)
        expert = self.expert(self.attn, self.hidden)
        for row in range(4):
            torch.testing.assert_close(result[row, [0, 4]], expert[row, [0, 4]])
            torch.testing.assert_close(result[row, [1, 2, 3, 5]], anchor[row, [1, 2, 3, 5]], rtol=0, atol=0)

    def test_disjoint_two_experts(self):
        second = PROCESSOR(8, rank=2)
        masks = torch.tensor([[[1., 0., 0.], [0., 0., 0.]], [[0., 1., 0.], [0., 0., 0.]]])
        result = self.route(masks, experts=[self.expert, second])(self.attn, self.hidden)
        torch.testing.assert_close(result[:, 0], self.expert(self.attn, self.hidden)[:, 0])
        torch.testing.assert_close(result[:, 1], second(self.attn, self.hidden)[:, 1])
        torch.testing.assert_close(result[:, 2:], self.anchor(self.attn, self.hidden)[:, 2:])

    def test_overlap_weights_are_normalized(self):
        masks = torch.ones(2, 4, 6)
        weights = normalized_weights(masks, [2., 1.], (2, 3))
        torch.testing.assert_close(weights.sum(dim=0), torch.ones(1, 2, 3))
        torch.testing.assert_close(weights[0], torch.full((1, 2, 3), 2 / 3))

    def test_soft_mask_resizing(self):
        mask = torch.zeros(1, 4, 6)
        mask[:, :2, :3] = 1
        weights = normalized_weights(mask, [0.5], (2, 3))
        self.assertEqual(weights.shape, (1, 1, 2, 3))
        self.assertLessEqual(weights.max(), 0.5)

    def test_cross_attention_expert_degeneracy(self):
        attn = SmallAttention(context=6)
        anchor, expert = PROCESSOR(8, 6, 2), PROCESSOR(8, 6, 2)
        context = torch.randn(4, 5, 6)
        router = RegionalAttnProcessor(anchor, [expert], torch.ones(1, 2, 3), [1.])
        router.spatial_shape = (2, 3)
        torch.testing.assert_close(router(attn, self.hidden, context), expert(attn, self.hidden, context))

    def test_both_branch_loras_and_saim_receive_training_gradients(self):
        target = torch.randn_like(self.hidden)
        loss = F.mse_loss(self.expert(self.attn, self.hidden), target)
        loss.backward()
        for branch in (0, 1):
            gradient = self.expert.to_q_lora[branch].up.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum(), 0)
        self.assertGreater(self.expert.temporal_q.weight.grad.abs().sum(), 0)

    def test_routing_broadcasts_across_multiple_sample_pairs(self):
        hidden = torch.randn(8, 6, 8)  # 2 samples x 2 frames x CFG
        router = self.route(torch.ones(1, 2, 3))
        torch.testing.assert_close(router(self.attn, hidden), self.expert(self.attn, hidden))

    def test_reject_invalid_layout_and_shape(self):
        with self.assertRaises(ValueError):
            self.route(torch.full((1, 2, 3), float("nan")))
        with self.assertRaises(ValueError):
            RegionalAttnProcessor(self.anchor, [self.expert], torch.ones(1, 2, 3), [-1.])
        router = self.route(torch.ones(1, 2, 3))
        router.spatial_shape = (3, 3)
        with self.assertRaises(ValueError):
            router(self.attn, self.hidden)

    def test_banks_install_hooks_restore_and_shared_saim(self):
        unet = SmallUNet()
        originals = dict(unet.attn_processors)
        name = next(iter(originals))
        with tempfile.TemporaryDirectory() as tmp:
            anchor_path, expert_path = Path(tmp) / "anchor.pt", Path(tmp) / "expert.pt"
            meta = dict(role="anchor", base_model="test", prompt_blend="a", prompt_fg="b")
            save_bank(anchor_path, {name: self.anchor}, meta)
            meta.update(role="expert", anchor_sha256=file_digest(anchor_path))
            save_bank(expert_path, {name: self.expert}, meta)
            state = read_bank(expert_path)["layers"][name]["state"]
            self.assertTrue(any(k.startswith("temporal_") for k in state))
            for branch in (0, 1):
                self.assertIn(f"to_q_lora.{branch}.up.weight", state)
            with patch("regional_adapters._processor_class", return_value=PROCESSOR):
                handle = install_regional_adapters(unet, anchor_path, [expert_path], torch.ones(1, 2, 3))
                pixels = self.hidden.transpose(1, 2).reshape(4, 8, 2, 3)
                result = unet.block(pixels)
                torch.testing.assert_close(result, self.expert(unet.block.transformer_blocks[0].attn1, self.hidden))
                handle.remove()
                self.assertIs(unet.attn_processors[name], originals[name])
                self.assertEqual(len(unet.block._forward_pre_hooks), 0)
                handle = install_regional_adapters(unet, anchor_path, [expert_path], torch.ones(1, 2, 3), shared_saim=True)
                routed_expert = unet.attn_processors[name].experts[0]
                torch.testing.assert_close(routed_expert.temporal_q.weight, self.anchor.temporal_q.weight)
                torch.testing.assert_close(routed_expert.to_q_lora[0].up.weight, self.expert.to_q_lora[0].up.weight)
                handle.remove()
            meta["anchor_sha256"] = "wrong"
            save_bank(expert_path, {name: self.expert}, meta)
            with self.assertRaises(ValueError):
                install_regional_adapters(unet, anchor_path, [expert_path], torch.ones(1, 2, 3))


class DatasetTests(unittest.TestCase):
    def test_export_user_normal_folder_with_source_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotated, normal = root / "abnormal", root / "normal"
            annotated.mkdir()
            normal.mkdir()
            Image.new("RGB", (32, 24), "gray").save(annotated / "defect.png")
            Image.new("RGB", (32, 24), "white").save(normal / "healthy.png")
            annotation = dict(imagePath="defect.png", imageWidth=32, imageHeight=24,
                              shapes=[dict(label="groove", shape_type="polygon",
                                           points=[[3, 3], [20, 3], [20, 8], [3, 8]])])
            (annotated / "defect.json").write_text(json.dumps(annotation))
            output = root / "output"
            subprocess.run([sys.executable, str(ROOT / "export_core_annotations.py"), str(annotated),
                            str(output), "--classes", "honeycomb", "groove", "--normal-images", str(normal),
                            "--reference-images"], check=True, capture_output=True)
            self.assertEqual(len(ManifestDataset(output / "normal.jsonl", "normal", normal=True)), 1)
            self.assertEqual(len(ManifestDataset(output / "groove.jsonl", "groove")), 1)
            self.assertFalse(list((output / "images").iterdir()))
            row = json.loads((output / "normal.jsonl").read_text())
            self.assertEqual(row["normal_label_source"], "user_provided_normal_directory")

    def test_synchronized_pair_and_normal_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixels = np.full((12, 24, 3), 127, dtype=np.uint8)
            pixels[2:8, 3:11] = 255
            Image.fromarray(pixels).save(root / "image.png")
            mask = np.zeros((12, 24), dtype=np.uint8)
            mask[2:8, 3:11] = 255
            Image.fromarray(mask).save(root / "mask.png")
            row = dict(image="image.png", mask="mask.png", **{"class": "honeycomb"}, labels=["honeycomb"])
            path = root / "data.jsonl"
            path.write_text(json.dumps(row) + "\n")
            dataset = ManifestDataset(path, "honeycomb", size=24)
            random.seed(2)
            sample = dataset[0]
            torch.testing.assert_close(sample["fg"], sample["blend"] * sample["mask"])
            self.assertEqual(sample["mask"].sum(), 48)
            row.update(labels=["honeycomb", "groove"])
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                ManifestDataset(path, "honeycomb", size=24)
            row.update(labels=[], **{"class": "normal"})
            path.write_text(json.dumps(row))
            sample = ManifestDataset(path, "normal", size=24, normal=True)[0]
            self.assertEqual(sample["fg"].abs().sum(), 0)

    def test_annotation_export_retains_overlap_but_excludes_mixed_expert_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "annotations"
            images.mkdir()
            Image.new("RGB", (24, 16), "gray").save(images / "sample.png")
            annotation = dict(imagePath="sample.png", imageWidth=24, imageHeight=16,
                              shapes=[dict(label=category, points=[[2, 2], [15, 2], [15, 12], [2, 12]],
                                           shape_type="polygon") for category in ["honeycomb", "groove"]])
            (images / "sample.json").write_text(json.dumps(annotation))
            output = root / "export"
            subprocess.run([sys.executable, str(ROOT / "export_core_annotations.py"), str(images), str(output)],
                           check=True, capture_output=True)
            self.assertFalse((output / "honeycomb.jsonl").exists())
            self.assertFalse((output / "groove.jsonl").exists())
            row = json.loads((output / "all_samples.jsonl").read_text())
            self.assertEqual(set(row["labels"]), {"honeycomb", "groove"})
            a = np.array(Image.open(output / row["masks"]["honeycomb"]))
            b = np.array(Image.open(output / row["masks"]["groove"]))
            self.assertTrue(np.any((a > 0) & (b > 0)))

    def test_random_layout_is_inside_core_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core = np.zeros((64, 96), dtype=np.uint8)
            core[15:45, 8:88] = 255
            Image.fromarray(core).save(root / "core.png")
            output = root / "layout"
            subprocess.run([sys.executable, str(ROOT / "make_regional_layout.py"), str(root / "core.png"), str(output)],
                           check=True, capture_output=True)
            a = np.array(Image.open(output / "region_0.png")) > 0
            b = np.array(Image.open(output / "region_1.png")) > 0
            self.assertTrue(a.any() and b.any())
            self.assertFalse((a & b).any())
            self.assertTrue(np.all(core[a | b] > 0))


class SamplerTests(unittest.TestCase):
    """Exercise sampler orchestration without representing a real DDIM/SD model."""
    @staticmethod
    def pipe():
        class VAE:
            config = SimpleNamespace(scaling_factor=1.)

            def encode(self, pixels):
                latents = F.avg_pool2d(pixels, 8)
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latents))

            def decode(self, latents):
                return SimpleNamespace(sample=F.interpolate(latents, scale_factor=8, mode="nearest"))

        class UNet:
            device = torch.device("cpu")
            dtype = torch.float32
            config = SimpleNamespace(in_channels=3)

            def __call__(self, inputs, timestep, encoder_hidden_states):
                self.last_embeddings = encoder_hidden_states.clone()
                return SimpleNamespace(sample=torch.ones_like(inputs) * 0.1)

        class Scheduler:
            init_noise_sigma = 1.

            def set_timesteps(self, steps, device):
                self.timesteps = torch.arange(steps - 1, -1, -1, device=device)

            def add_noise(self, original, noise, times):
                return original + noise * (times.float().reshape(-1, 1, 1, 1) + 1) * 0.1

            def scale_model_input(self, inputs, time):
                return inputs

            def step(self, prediction, time, latents, generator):
                return SimpleNamespace(prev_sample=latents - prediction)

        def encode(prompt, device, count, cfg):
            value = 1. if prompt == "whole" else 2.
            return torch.tensor([0., value] if cfg else [value]).reshape(-1, 1, 1)

        return SimpleNamespace(unet=UNet(), vae=VAE(), scheduler=Scheduler(), vae_scale_factor=8,
                               _encode_prompt=encode, progress_bar=lambda times: times)

    def test_generation_seed_and_cfg_branch_embedding_order(self):
        pipe = self.pipe()
        metadata = dict(prompt_blend="whole", prompt_fg="foreground")
        masks = torch.zeros(1, 16, 24)
        a = sample(pipe, metadata, masks, 16, 24, seed=2, steps=4)
        b = sample(pipe, metadata, masks, 16, 24, seed=2, steps=4)
        self.assertEqual(a[0].size, (24, 16))
        np.testing.assert_array_equal(np.array(a[0]), np.array(b[0]))
        torch.testing.assert_close(pipe.unet.last_embeddings.flatten(), torch.tensor([0., 0., 1., 2.]))
        sample(pipe, metadata, masks, 16, 24, seed=2, steps=4, guidance=1.)
        torch.testing.assert_close(pipe.unet.last_embeddings.flatten(), torch.tensor([1., 2.]))

    def test_zero_mask_edit_restores_reference_latents(self):
        pipe = self.pipe()
        reference = Image.new("RGB", (24, 16), (100, 150, 200))
        image, _ = sample(pipe, dict(prompt_blend="whole", prompt_fg="foreground"),
                          torch.zeros(1, 16, 24), 16, 24, steps=4, reference=reference, strength=0.5)
        np.testing.assert_array_equal(np.array(image), np.array(reference))


if __name__ == "__main__":
    unittest.main()
