"""Numerical references used by tests and development benchmarks."""

from __future__ import annotations

import torch

from comfy_kitchen.backends._activations import apply_input_act
from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation


def _prepare_activation(
    x: torch.Tensor,
    *,
    convrot: bool,
    convrot_groupsize: int,
    input_act: str | None,
) -> torch.Tensor:
    x = apply_input_act(x, input_act)
    if convrot:
        h = _build_hadamard(
            convrot_groupsize,
            device=x.device,
            dtype=x.dtype,
        )
        x = _rotate_activation(x, h, convrot_groupsize)
    return x


def reference_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    activation_mode: str = "dynamic",
) -> torch.Tensor:
    """Float32 CPU-style reference for either supported activation mode."""

    activated = _prepare_activation(
        x,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
        input_act=input_act,
    )
    original_shape = activated.shape
    x_2d = activated.reshape(-1, activated.shape[-1])

    if activation_mode == "dynamic":
        quantized_x, row_scale = quantize_int8_rowwise(x_2d)
        lhs = quantized_x.float()
    elif activation_mode == "weight_only":
        lhs = x_2d.float()
        row_scale = None
    else:
        raise ValueError(f"Unsupported activation mode: {activation_mode!r}")

    scales = weight_scale.float().reshape(-1)
    if scales.numel() == 1:
        scales = scales.expand(weight.shape[0])
    if scales.numel() != weight.shape[0]:
        raise ValueError("Invalid weight scale shape")

    result = lhs @ weight.float().T
    result.mul_(scales.reshape(1, -1))
    if row_scale is not None:
        result.mul_(row_scale.float())
    if bias is not None:
        result.add_(bias.float().reshape(1, -1))
    return result.to(out_dtype).reshape(*original_shape[:-1], weight.shape[0])
