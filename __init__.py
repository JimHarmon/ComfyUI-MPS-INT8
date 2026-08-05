"""ComfyUI entry point for the MPS INT8 backend extension."""

from .comfyui_mps_int8 import backend_status_text, register_backend
from .comfyui_mps_int8.status_node import MPSINT8BackendStatus

# Registration is idempotent. Import-time registration is intentional because
# ComfyUI imports custom-node packages before workflows execute.
register_backend()

NODE_CLASS_MAPPINGS = {
    "MPSINT8BackendStatus": MPSINT8BackendStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MPSINT8BackendStatus": "MPS INT8 Backend Status",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "backend_status_text",
]
