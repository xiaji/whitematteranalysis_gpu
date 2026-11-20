# -*- coding: utf-8 -*-

"""GPU memory helpers: OOM detection, TF32 policy, and VRAM budget capping."""

import math

from .config import config

_OOM_SUBSTRINGS = (
    'out of memory',
    'cublas_status_alloc_failed',
    'cudnn_status_alloc_failed',
    'cuda error: out of memory',
)


def is_gpu_oom(exc: BaseException) -> bool:
    """Return True if *exc* represents a CUDA/cuBLAS/cuDNN/MPS allocation failure."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(token in msg for token in _OOM_SUBSTRINGS)
    return False


def disable_tf32() -> None:
    """Disable TF32 on CUDA matmul and cuDNN to preserve IEEE float32/64 parity."""
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = False


def _device_type(device) -> str:
    if device is None:
        return 'cuda'
    if hasattr(device, 'type'):
        return device.type
    return str(device).split(':', 1)[0]


def empty_cache(device=None) -> None:
    """Release cached allocator memory for *device* without calling CUDA on MPS/CPU."""
    try:
        import torch
    except ImportError:
        return
    dev_type = _device_type(device)
    if dev_type == 'cuda':
        torch.cuda.empty_cache()
    elif dev_type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def effective_free_vram(device) -> int:
    """Return free VRAM bytes for *device*, capped by ``config.max_vram_gib``."""
    free_bytes = 4 << 30
    if _device_type(device) != 'mps':
        try:
            import torch
            free_bytes, _total = torch.cuda.mem_get_info(device)
        except Exception:
            pass
    cap_gib = config.max_vram_gib
    if cap_gib is not None and math.isfinite(cap_gib) and cap_gib > 0:
        free_bytes = min(int(free_bytes), int(cap_gib * (1 << 30)))
    return int(free_bytes)
