#!/usr/bin/env python3
"""Microbenchmark the native MPS INT8 backend on representative shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from comfyui_mps_int8.backend import _int8_linear_impl, _has_native_mps_kernel  # noqa: E402


SHAPES = {
    "small": (96, 64),
    "h3_qkv": (21504, 5376),
    "h3_out": (5376, 7168),
    "h3_fc1": (28672, 5376),
    "h3_fc2": (5376, 14336),
}


def mib(value: int) -> float:
    return value / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=SHAPES, default="small")
    parser.add_argument("--rows", type=int, default=8, help="Activation rows M")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--activation-mode",
        choices=("dynamic", "weight_only"),
        default="dynamic",
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available() or not _has_native_mps_kernel():
        raise SystemExit("MPS native INT8 kernel is unavailable")

    dtype = getattr(torch, args.dtype)
    out_features, in_features = SHAPES[args.shape]
    torch.manual_seed(2026)

    x = torch.randn(args.rows, in_features, dtype=dtype, device="mps")
    weight = torch.randint(
        -127,
        128,
        (out_features, in_features),
        dtype=torch.int8,
        device="mps",
    )
    scale = (
        torch.rand(out_features, 1, dtype=torch.float32, device="mps")
        * 0.01
        + 0.001
    )

    def invoke():
        return _int8_linear_impl(
            x,
            weight,
            scale,
            out_dtype=dtype,
            convrot=False,
            input_act=None,
            activation_mode=args.activation_mode,
        )

    for _ in range(args.warmup):
        invoke()
    torch.mps.synchronize()

    before_current = torch.mps.current_allocated_memory()
    before_driver = torch.mps.driver_allocated_memory()
    started = time.perf_counter()
    output = None
    for _ in range(args.iterations):
        output = invoke()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    after_current = torch.mps.current_allocated_memory()
    after_driver = torch.mps.driver_allocated_memory()

    assert output is not None
    finite_fraction = torch.isfinite(output).float().mean().item()
    report = {
        "shape_name": args.shape,
        "m": args.rows,
        "n": out_features,
        "k": in_features,
        "dtype": args.dtype,
        "activation_mode": args.activation_mode,
        "iterations": args.iterations,
        "mean_seconds": elapsed / args.iterations,
        "finite_fraction": finite_fraction,
        "current_memory_before_mib": mib(before_current),
        "current_memory_after_mib": mib(after_current),
        "driver_memory_before_mib": mib(before_driver),
        "driver_memory_after_mib": mib(after_driver),
    }
    print(json.dumps(report, indent=2))
    return 0 if finite_fraction == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
