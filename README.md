# ComfyUI-MPS-INT8

`ComfyUI-MPS-INT8` adds an Apple Silicon MPS backend for
`comfy_kitchen.int8_linear`. It enables compatible INT8 ConvRot models, such
as MiniMax H3, to run without relying on the unsupported MPS
`aten::_int_mm` operation.

The extension operates below the workflow layer. Existing ComfyUI workflows
and sampler nodes do not need to be modified.

## Requirements

- An Apple Silicon Mac with MPS support
- ComfyUI with `comfy-kitchen` quantized model support
- A PyTorch build that provides an MPS kernel for
  `aten::_weight_int8pack_mm`

## Installation

Place the repository in ComfyUI's custom-node directory, e.g.:

```text
~/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/ComfyUI-MPS-INT8/
```

Restart ComfyUI after installation. When the required runtime support is
available, the extension registers `mps_int8_native` ahead of the eager
`comfy-kitchen` backend.

## Activation modes

The activation mode can be selected before starting ComfyUI:

```bash
export COMFY_MPS_INT8_ACTIVATION_MODE=dynamic
```

Available modes:

- `dynamic` is the default. It retains row-wise activation quantization before
  invoking the native weight-INT8 operation.
- `weight_only` skips dynamic activation quantization and uses fewer activation
  temporaries. Its numerical behavior differs from the eager INT8 path.

Both modes keep checkpoint weights in INT8.

## Implementation

The backend uses PyTorch's native MPS operator:

```text
aten::_weight_int8pack_mm
```

This operator performs the dequantizing multiplication in its Metal kernel,
without materializing a complete BF16 or FP32 copy of the model weights. The
extension also handles the optional input activation, ConvRot transformation,
weight scales, output conversion, and bias required by
`comfy_kitchen.int8_linear`.

The native operator requires both dimensions of each two-dimensional INT8
weight tensor to be divisible by 32. Unsupported shapes are rejected rather
than passed to the unavailable MPS `_int_mm` path.
