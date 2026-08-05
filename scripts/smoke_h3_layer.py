#!/usr/bin/env python3
"""Run one real H3 INT8 layer through public comfy-kitchen dispatch on MPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from safetensors import safe_open
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import comfy_kitchen  # noqa: E402
from comfy_kitchen.registry import registry  # noqa: E402
from comfyui_mps_int8.backend import (  # noqa: E402
    BACKEND_NAME,
    backend_status,
    register_backend,
)


DEFAULT_LAYER = "blocks.0.mlp.fc2"


def mib(value: int) -> float:
    return value / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument(
        "--input-act",
        choices=("none", "gelu_tanh", "swiglu"),
        default="swiglu",
    )
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    args = parser.parse_args()

    if not register_backend():
        raise SystemExit(json.dumps(backend_status(), indent=2))

    weight_key = f"{args.layer}.weight"
    scale_key = f"{args.layer}.weight_scale"
    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = [key for key in (weight_key, scale_key) if key not in keys]
        if missing:
            raise SystemExit(f"Missing checkpoint tensors: {missing}")
        weight = handle.get_tensor(weight_key)
        weight_scale = handle.get_tensor(scale_key)

    out_features, in_features = weight.shape
    raw_features = in_features * 2 if args.input_act == "swiglu" else in_features
    input_act = None if args.input_act == "none" else args.input_act
    x = torch.randn(
        args.rows,
        raw_features,
        dtype=torch.bfloat16,
        device="mps",
    )

    kwargs = {
        "x": x,
        "weight": weight,
        "weight_scale": weight_scale,
        "bias": None,
        "out_dtype": torch.bfloat16,
        "convrot": True,
        "convrot_groupsize": args.convrot_groupsize,
        "input_act": input_act,
    }
    selected = registry.get_capable_backend("int8_linear", kwargs=kwargs)
    before_current = torch.mps.current_allocated_memory()
    before_driver = torch.mps.driver_allocated_memory()
    started = time.perf_counter()
    output = comfy_kitchen.int8_linear(**kwargs)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    after_current = torch.mps.current_allocated_memory()
    after_driver = torch.mps.driver_allocated_memory()

    report = {
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "weight_shape": list(weight.shape),
        "raw_input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "input_act": input_act,
        "selected_backend": selected,
        "expected_backend": BACKEND_NAME,
        "elapsed_seconds": elapsed,
        "finite_fraction": torch.isfinite(output).float().mean().item(),
        "current_memory_before_mib": mib(before_current),
        "current_memory_after_mib": mib(after_current),
        "driver_memory_before_mib": mib(before_driver),
        "driver_memory_after_mib": mib(after_driver),
    }
    print(json.dumps(report, indent=2))
    return 0 if selected == BACKEND_NAME and report["finite_fraction"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
