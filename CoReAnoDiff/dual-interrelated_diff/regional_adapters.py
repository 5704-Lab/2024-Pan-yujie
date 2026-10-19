"""Complete dual-branch adapter banks and spatial attention-output routing.

No changes to the vendored attention processor are needed. Each expert includes
both branch LoRAs and its SAIM parameters; routing is applied after that processor.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _processor_class():
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0
    return LoRAAttnProcessor2_0


def create_processors(unet, rank=4, saim_init="zero"):
    if saim_init not in ("zero", "legacy_random"):
        raise ValueError("saim_init must be zero or legacy_random")
    cls = _processor_class()
    processors = {}
    for name in unet.attn_processors:
        attention = unet.get_submodule(name[:-len(".processor")])
        processors[name] = cls(
            hidden_size=attention.to_q.in_features,
            cross_attention_dim=None if name.endswith("attn1.processor") else attention.to_k.in_features,
            rank=rank,
        )
        if saim_init == "zero":
            # LoRA already starts with zero output. SAIM must also initially be
            # a zero residual so fresh adapters preserve the pretrained UNet.
            # Keep upstream Q/K/V/I random so temporal_o learns on the first
            # update and the other temporal parameters receive gradients later.
            nn.init.zeros_(processors[name].temporal_o.weight)
            nn.init.zeros_(processors[name].temporal_o.bias)
    return processors


def save_bank(path, processors, metadata):
    """Serialize state_dicts verbatim, including temporal_* and both ModuleLists."""
    layers = {}
    for name, processor in processors.items():
        layers[name] = {
            "hidden_size": processor.hidden_size,
            "cross_attention_dim": processor.cross_attention_dim,
            "rank": processor.rank,
            "state": {k: v.detach().cpu().clone() for k, v in processor.state_dict().items()},
        }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"format": "dualanodiff-regional-v1", "metadata": metadata, "layers": layers}, temporary)
    temporary.replace(target)


def read_bank(path):
    bank = torch.load(path, map_location="cpu", weights_only=True)
    if bank.get("format") != "dualanodiff-regional-v1" or not bank.get("layers"):
        raise ValueError(f"Unsupported or empty adapter bank: {path}")
    return bank


def processors_from_bank(bank, unet, shared_saim=None):
    if set(bank["layers"]) != set(unet.attn_processors):
        raise ValueError("Adapter and UNet attention layer names do not match")
    cls = _processor_class()
    processors = {}
    for name, layer in bank["layers"].items():
        processor = cls(layer["hidden_size"], layer["cross_attention_dim"], layer["rank"])
        state = dict(layer["state"])
        if shared_saim is not None:
            state.update({k: v for k, v in shared_saim["layers"][name]["state"].items()
                          if k.startswith("temporal_")})
        processor.load_state_dict(state, strict=True)
        attention = unet.get_submodule(name[:-len(".processor")])
        if attention.to_q.in_features != layer["hidden_size"]:
            raise ValueError(f"Hidden dimension mismatch: {name}")
        processor.to(device=attention.to_q.weight.device, dtype=attention.to_q.weight.dtype)
        processors[name] = processor
    return processors


def normalized_weights(masks, strengths, spatial_shape, mode="bilinear"):
    """K x H x W -> K x 1 x h x w; overlaps have total coefficient <= 1."""
    if mode not in ("nearest", "bilinear"):
        raise ValueError("mask interpolation must be nearest or bilinear")
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    weights = F.interpolate(masks[:, None].float(), size=spatial_shape, mode=mode, **kwargs)
    weights = weights * torch.as_tensor(strengths, device=masks.device)[:, None, None, None]
    return weights / weights.sum(dim=0, keepdim=True).clamp_min(1.0)


class RegionalAttnProcessor(nn.Module):
    def __init__(self, anchor, experts, masks, strengths, interpolation="bilinear"):
        super().__init__()
        masks = torch.as_tensor(masks, dtype=torch.float32)
        if masks.ndim != 3 or masks.shape[0] != len(experts) or not len(experts):
            raise ValueError("Expected one HxW mask per expert")
        if not torch.isfinite(masks).all() or (masks < 0).any() or (masks > 1).any():
            raise ValueError("Masks must be finite values in [0, 1]")
        if len(strengths) != len(experts) or any(not math.isfinite(x) or x < 0 for x in strengths):
            raise ValueError("Expected one finite nonnegative strength per expert")
        self.anchor = anchor
        self.experts = nn.ModuleList(experts)
        self.register_buffer("masks", masks)
        self.strengths = tuple(strengths)
        self.interpolation = interpolation
        self.spatial_shape = None

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        # The original processor rearranges (b, frame) and expects two interleaved
        # frames. CFG repeats the complete batch, preserving this ordering.
        if hidden_states.ndim != 3 or hidden_states.shape[0] % 2:
            raise ValueError("Regional routing requires interleaved whole/foreground token batches")
        if self.spatial_shape is None or math.prod(self.spatial_shape) != hidden_states.shape[1]:
            raise ValueError("Missing or inconsistent Transformer2D spatial shape")
        kwargs = dict(encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, scale=scale)
        base = self.anchor(attn, hidden_states, **kwargs)
        weights = normalized_weights(self.masks, self.strengths, self.spatial_shape, self.interpolation)
        weights = weights.to(device=base.device, dtype=base.dtype)
        result = base
        for index, expert in enumerate(self.experts):
            if not weights[index].any():
                continue
            candidate = expert(attn, hidden_states, **kwargs)
            spatial_weight = weights[index].reshape(1, -1, 1)
            result = result + spatial_weight * (candidate - base)
        return result


class RoutingHandle:
    def __init__(self, unet, originals, hooks):
        self.unet, self.originals, self.hooks = unet, originals, hooks

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.unet.set_attn_processor(dict(self.originals))
        self.hooks = []


def install_regional_adapters(unet, anchor_path, expert_paths, masks, strengths=None,
                              interpolation="bilinear", shared_saim=False):
    anchor_bank = read_bank(anchor_path)
    if anchor_bank["metadata"].get("role") != "anchor":
        raise ValueError("The baseline bank must be a normal-core anchor")
    experts = [read_bank(path) for path in expert_paths]
    if not experts:
        raise ValueError("At least one expert is required")
    for bank in experts:
        if bank["metadata"].get("anchor_sha256") != file_digest(anchor_path):
            raise ValueError("All experts must be trained from the supplied anchor")
        for key in ("base_model", "prompt_blend", "prompt_fg"):
            if bank["metadata"].get(key) != anchor_bank["metadata"].get(key):
                raise ValueError(f"Incompatible bank metadata: {key}")
    anchors = processors_from_bank(anchor_bank, unet)
    expert_layers = [processors_from_bank(bank, unet, anchor_bank if shared_saim else None) for bank in experts]
    originals = dict(unet.attn_processors)
    routed = {
        name: RegionalAttnProcessor(anchor, [layers[name] for layers in expert_layers], masks,
                                    strengths if strengths is not None else [1.0] * len(experts), interpolation)
        for name, anchor in anchors.items()
    }
    # Every Transformer2D sees BCHW before flattening. Record its actual shape,
    # including rectangular feature maps; never infer height by sqrt(tokens).
    groups = {}
    for name, processor in routed.items():
        parent_name = name.split(".transformer_blocks.")[0]
        groups.setdefault(parent_name, []).append(processor)
    hooks = []
    try:
        for name, processors in groups.items():
            def capture(module, inputs, processors=processors):
                if not inputs or inputs[0].ndim != 4:
                    raise ValueError("Expected BCHW Transformer2D input")
                for processor in processors:
                    processor.spatial_shape = inputs[0].shape[-2:]
            hooks.append(unet.get_submodule(name).register_forward_pre_hook(capture))
        unet.set_attn_processor(dict(routed))
    except Exception:
        for hook in hooks:
            hook.remove()
        unet.set_attn_processor(dict(originals))
        raise
    return RoutingHandle(unet, originals, hooks)
