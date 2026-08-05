from __future__ import annotations

import unittest

import torch

from comfyui_mps_int8.backend import (
    _normalize_weight_scale,
    _validate_native_shapes,
)
from comfyui_mps_int8.reference import reference_int8_linear


class ScaleContractTests(unittest.TestCase):
    def test_scalar_scale_expands_and_matches_dtype(self):
        scale = torch.tensor([[0.25]], dtype=torch.float32)
        result = _normalize_weight_scale(
            scale,
            64,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )
        self.assertEqual(result.shape, (64,))
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(result.is_contiguous())
        self.assertTrue(torch.all(result == torch.tensor(0.25, dtype=torch.float16)))

    def test_per_output_scale_flattens(self):
        scale = torch.arange(32, dtype=torch.float32).reshape(32, 1)
        result = _normalize_weight_scale(
            scale,
            32,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, (32,))
        torch.testing.assert_close(result, torch.arange(32, dtype=torch.float32))

    def test_invalid_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scalar or per-output-channel"):
            _normalize_weight_scale(
                torch.ones(3),
                32,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


class ShapeContractTests(unittest.TestCase):
    def test_known_h3_shapes_are_supported(self):
        shapes = (
            (5376, 7168),
            (21504, 5376),
            (28672, 5376),
            (5376, 14336),
        )
        for out_features, in_features in shapes:
            x = torch.empty(1, in_features)
            weight = torch.empty(out_features, in_features, dtype=torch.int8)
            _validate_native_shapes(x, weight)

    def test_non_multiple_of_32_fails_closed(self):
        x = torch.empty(2, 65)
        weight = torch.empty(96, 65, dtype=torch.int8)
        with self.assertRaisesRegex(RuntimeError, "divisible by 32"):
            _validate_native_shapes(x, weight)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.x = torch.randn(3, 64, dtype=torch.float32)
        self.weight = torch.randint(-20, 21, (96, 64), dtype=torch.int8)
        self.scale = torch.rand(96, 1, dtype=torch.float32) * 0.01 + 0.001
        self.bias = torch.randn(96, dtype=torch.float32) * 0.01

    def test_weight_only_reference_shape(self):
        result = reference_int8_linear(
            self.x,
            self.weight,
            self.scale,
            self.bias,
            out_dtype=torch.float32,
            activation_mode="weight_only",
        )
        self.assertEqual(result.shape, (3, 96))
        self.assertTrue(torch.isfinite(result).all())

    def test_dynamic_reference_with_convrot_and_activation(self):
        result = reference_int8_linear(
            self.x,
            self.weight,
            self.scale,
            self.bias,
            out_dtype=torch.float32,
            convrot=True,
            convrot_groupsize=16,
            input_act="gelu_tanh",
            activation_mode="dynamic",
        )
        self.assertEqual(result.shape, (3, 96))
        self.assertTrue(torch.isfinite(result).all())

    def test_dynamic_reference_with_h3_swiglu_width_change(self):
        swiglu_input = torch.randn(3, 128, dtype=torch.float32)
        result = reference_int8_linear(
            swiglu_input,
            self.weight,
            self.scale,
            self.bias,
            out_dtype=torch.float32,
            convrot=True,
            convrot_groupsize=16,
            input_act="swiglu",
            activation_mode="dynamic",
        )
        self.assertEqual(result.shape, (3, 96))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
