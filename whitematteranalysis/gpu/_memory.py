# -*- coding: utf-8 -*-

"""GPU memory helpers: OOM detection, TF32 policy, and VRAM budget capping."""

from .config import config

_OOM_SUBSTRINGS = (
    'out of memory',
    'cublas_status_alloc_failed',
    'cudnn_status_alloc_failed',
    'cuda error: out of memory',
)


def is_gpu_oom(exc: BaseException) -> bool:
    """Return True if *exc* represents a CUDA/cuBLAS/cuDNN allocation failure."""
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


def effective_free_vram(device) -> int:
    """Return free VRAM bytes for *device*, capped by ``config.max_vram_gib``."""
    free_bytes = 4 << 30
    try:
        import torch
        free_bytes, _total = torch.cuda.mem_get_info(device)
    except Exception:
        pass
    cap_gib = config.max_vram_gib
    if cap_gib is not None and cap_gib > 0:
        free_bytes = min(int(free_bytes), int(cap_gib * (1 << 30)))
    return int(free_bytes)
