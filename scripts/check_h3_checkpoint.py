#!/usr/bin/env python3
"""Inspect an H3 safetensors header without deserializing model tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    bad_shapes = []
    missing_scales = []
    int8_weights = []

    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for key in sorted(keys):
            tensor = handle.get_slice(key)
            shape = tuple(tensor.get_shape())
            if tensor.get_dtype() != "I8" or len(shape) != 2:
                continue
            int8_weights.append({"key": key, "shape": shape})
            if shape[0] % 32 or shape[1] % 32:
                bad_shapes.append({"key": key, "shape": shape})
            scale_key = f"{key}_scale"
            if scale_key not in keys:
                missing_scales.append(scale_key)

    report = {
        "checkpoint": str(args.checkpoint),
        "int8_weight_count": len(int8_weights),
        "all_shapes_native_mps_compatible": not bad_shapes,
        "bad_shapes": bad_shapes,
        "missing_scales": missing_scales,
        "largest_weights": sorted(
            int8_weights,
            key=lambda item: item["shape"][0] * item["shape"][1],
            reverse=True,
        )[:8],
    }
    print(json.dumps(report, indent=2))
    return 0 if not bad_shapes and not missing_scales else 1


if __name__ == "__main__":
    raise SystemExit(main())
