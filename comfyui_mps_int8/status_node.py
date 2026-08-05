"""A small diagnostic node; inference acceleration is registered globally."""

from __future__ import annotations

from .backend import backend_status_text


class MPSINT8BackendStatus:
    """Return the current registration and dispatch status as text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "report"
    CATEGORY = "MPS INT8/diagnostics"
    OUTPUT_NODE = True

    def report(self):
        return (backend_status_text(),)
