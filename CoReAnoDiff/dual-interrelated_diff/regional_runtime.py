"""Resolve the vendored dual-branch diffusers, never a site-packages substitute."""
from pathlib import Path


def check_local_diffusers():
    import diffusers
    expected = Path(__file__).resolve().parent / "diffusers"
    actual = Path(diffusers.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError(f"Expected repository diffusers at {expected}, imported {actual}")
    return diffusers
