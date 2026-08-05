from __future__ import annotations

import unittest

import torch

import comfy_kitchen
from comfy_kitchen.registry import registry
from comfyui_mps_int8.backend import (
    BACKEND_NAME,
    _has_native_mps_kernel,
    _int8_linear_impl,
    int8_linear,
    register_backend,
)
from comfyui_mps_int8.reference import reference_int8_linear


MPS_READY = bool(
    torch.backends.mps.is_available() and _has_native_mps_kernel()
)


@unittest.skipUnless(MPS_READY, "MPS native INT8 kernel is unavailable")
class NativeMPSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)
        self.x_cpu = torch.randn(7, 64, dtype=torch.bfloat16)
        self.weight_cpu = torch.randint(-30, 31, (96, 64), dtype=torch.int8)
        self.scale_cpu = torch.rand(96, 1, dtype=torch.float32) * 0.01 + 0.001
        self.bias_cpu = torch.randn(96, dtype=torch.float32) * 0.01

    def _run_and_compare(self, mode: str, **kwargs):
        actual = _int8_linear_impl(
            self.x_cpu.to("mps"),
            self.weight_cpu,
            self.scale_cpu,
            self.bias_cpu,
            out_dtype=torch.bfloat16,
            activation_mode=mode,
            **kwargs,
        )
        torch.mps.synchronize()
        reference = reference_int8_linear(
            self.x_cpu.float(),
            self.weight_cpu,
            self.scale_cpu,
            self.bias_cpu,
            out_dtype=torch.bfloat16,
            activation_mode=mode,
            **kwargs,
        )
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(
            actual.cpu().float(),
            reference.float(),
            rtol=3e-2,
            atol=3e-2,
        )

    def test_weight_only_matches_reference(self):
        self._run_and_compare(
            "weight_only",
            convrot=False,
            convrot_groupsize=16,
            input_act=None,
        )

    def test_dynamic_matches_reference_with_convrot_and_gelu(self):
        self._run_and_compare(
            "dynamic",
            convrot=True,
            convrot_groupsize=16,
            input_act="gelu_tanh",
        )

    def test_dynamic_matches_reference_with_h3_swiglu(self):
        self.x_cpu = torch.randn(7, 128, dtype=torch.bfloat16)
        self._run_and_compare(
            "dynamic",
            convrot=True,
            convrot_groupsize=16,
            input_act="swiglu",
        )

    def test_registry_selects_backend(self):
        self.assertTrue(register_backend())
        kwargs = {
            "x": self.x_cpu.to("mps"),
            "weight": self.weight_cpu,
            "weight_scale": self.scale_cpu,
            "bias": self.bias_cpu,
            "out_dtype": torch.bfloat16,
            "convrot": False,
            "convrot_groupsize": 16,
            "input_act": None,
        }
        selected = registry.get_capable_backend("int8_linear", kwargs=kwargs)
        implementation = registry.get_implementation("int8_linear", kwargs=kwargs)
        self.assertEqual(selected, BACKEND_NAME)
        self.assertIs(implementation, int8_linear)

        result = comfy_kitchen.int8_linear(**kwargs)
        torch.mps.synchronize()
        self.assertEqual(result.shape, (7, 96))
        self.assertTrue(torch.isfinite(result).all().item())


if __name__ == "__main__":
    unittest.main()
