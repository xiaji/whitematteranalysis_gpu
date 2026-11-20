# -*- coding: utf-8 -*-

""" gpu.config

Global configuration for GPU acceleration in whitematteranalysis.

The configuration can be set programmatically (e.g. from command line
arguments in bin/ scripts) or through environment variables:

    WMA_GPU           'auto' (default) | '1'/'on' | '0'/'off'
    WMA_GPU_FLOAT32   '1' to compute distances in float32 (fast mode).
                      Default '0' (float64, matches CPU results).
    WMA_GPU_DEVICES   comma separated device ids to use, e.g. "0,1".
                      Default: all visible CUDA devices.
    WMA_GPU_TILE      override the number of query fibers per GPU tile.
    WMA_GPU_MIN_PAIR_WORK
                      minimum query*fixed pair count to dispatch to CUDA
                      (default 4096; smaller clusters use cpu_backend).
    WMA_GPU_MAX_VRAM_GIB
                      optional upper bound (GiB) on free VRAM assumed by
                      GPU tile sizing (default: unset, use full free VRAM).
    WMA_GPU_STRICT    '1' to force bit-exact per-pair registration reduction
                      parity with eager PyTorch (default '0', fused 2D kernel).
    WMA_GPU_MPS       '1' to opt in to Apple Silicon MPS eager float32 when
                      CUDA is unavailable and float32 is enabled (default '0').

"""

import math
import os


class GPUConfig:
    """Process-global GPU settings. See module docstring for env vars."""

    def __init__(self):
        # None means auto (use GPU if torch with CUDA is importable).
        self.requested = None
        # float32 fast mode. Default False: float64, matches CPU results.
        self.float32 = False
        # Strict registration reduction parity mode (WMA_GPU_STRICT).
        self.strict = False
        # Opt-in Apple Silicon MPS backend (WMA_GPU_MPS, requires float32).
        self.allow_mps = False
        # List of CUDA device indices to use, or None for all visible ones.
        self.devices = None
        # Override for the number of query fibers per computation tile.
        self.tile_rows = None
        # Optional cap (in GiB) on free VRAM used for GPU tile sizing.
        self.max_vram_gib = None
        # Minimum query*fixed pair count to dispatch to CUDA; smaller
        # clusters route to cpu_backend to avoid GPU launch overhead.
        self.min_pair_work = 4096
        # CUDA device index used by THIS process (single-GPU mode: 0,
        # or the first entry of WMA_GPU_DEVICES).
        self.device_index = 0
        # CPU accelerated backend (vectorized numpy, threaded). When GPU
        # is off, distances use this instead of the original per-fiber
        # joblib loops. Disable with WMA_CPU_FAST=0 for the original path.
        self.cpu_fast = os.environ.get('WMA_CPU_FAST', '1') in ('1', 'true', 'yes', 'on')
        # Worker threads for the CPU backend (WMA_CPU_THREADS).
        self.cpu_threads = int(os.environ.get('WMA_CPU_THREADS', '0') or 0)

    def cuda_available(self):
        """True if torch imports and at least one CUDA device works."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def mps_available(self):
        """True if torch imports and Apple Silicon MPS is available."""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except Exception:
            return False

    def cpu_fast_enabled(self):
        return bool(self.cpu_fast)

    def set_cpu_fast(self, on):
        self.cpu_fast = bool(on)


config = GPUConfig()


def set_gpu(on):
    """Set GPU usage. Accepts True/False/None or strings 'on'/'off'/'auto'."""
    if isinstance(on, str):
        s = on.strip().lower()
        if s in ('1', 'true', 'on', 'yes'):
            config.requested = True
        elif s in ('0', 'false', 'off', 'no'):
            config.requested = False
        else:
            config.requested = None
    else:
        config.requested = on


def set_float32(on):
    config.float32 = bool(on)


def set_strict(on):
    config.strict = bool(on)


def strict_enabled(requested=None):
    return config.strict if requested is None else bool(requested)


def mps_available():
    return config.mps_available()


def _mps_active():
    return (not config.cuda_available()
            and bool(config.allow_mps and config.float32)
            and mps_available())


def set_devices(devices):
    """Set the list of CUDA device indices to use (int list or '0,1' string)."""
    if isinstance(devices, str):
        devices = [int(d) for d in devices.replace(',', ' ').split() if d != '']
    if devices is not None and len(devices) == 0:
        devices = None
    config.devices = devices


def set_tile_rows(tile_rows):
    if tile_rows is not None:
        config.tile_rows = int(tile_rows)


def use_gpu(requested=None):
    """Resolve the tri-state GPU request against availability.

    requested: None (follow global config), True, False, or string.
    Global config default is None = auto, meaning use GPU when available.
    """
    r = config.requested if requested is None else requested
    if isinstance(r, str):
        if r.strip().lower() in ('0', 'false', 'off', 'no'):
            return False
        r = None if r.strip().lower() in ('auto', '') else True
    if r is False:
        return False
    available = config.cuda_available()
    if not available and config.allow_mps and config.float32:
        available = mps_available()
    if r is True and not available:
        raise RuntimeError(
            "GPU acceleration was explicitly requested, but CUDA is unavailable. "
            "Use -gpu auto or -gpu off to run without CUDA.")
    return available


def float32_enabled(requested=None):
    return config.float32 if requested is None else bool(requested)


def torch_module():
    """Import and return torch, raising a clear error if unavailable."""
    import torch
    return torch


def get_device(index=None):
    """Return the torch device this process should compute on."""
    torch = torch_module()
    if _mps_active():
        return torch.device('mps')
    idx = config.device_index if index is None else index
    if config.devices:
        idx = config.devices[idx % len(config.devices)]
    return torch.device('cuda', idx)


def visible_device_count():
    torch = torch_module()
    if _mps_active():
        return 1
    if config.devices:
        return len(config.devices)
    return torch.cuda.device_count()


def _from_env():
    set_gpu(os.environ.get('WMA_GPU', 'auto'))
    set_float32(os.environ.get('WMA_GPU_FLOAT32', '0').strip().lower() in ('1', 'true', 'yes', 'on'))
    set_strict(os.environ.get('WMA_GPU_STRICT', '0').strip().lower() in ('1', 'true', 'yes', 'on'))
    config.allow_mps = os.environ.get('WMA_GPU_MPS', '0').strip().lower() in ('1', 'true', 'yes', 'on')
    devices = os.environ.get('WMA_GPU_DEVICES')
    if devices:
        set_devices(devices)
    tile = os.environ.get('WMA_GPU_TILE')
    if tile:
        try:
            config.tile_rows = int(tile)
        except ValueError:
            pass
    min_pair_work = os.environ.get('WMA_GPU_MIN_PAIR_WORK')
    if min_pair_work is not None and min_pair_work != '':
        try:
            value = int(min_pair_work)
            if value >= 0:
                config.min_pair_work = value
        except ValueError:
            pass
    max_vram = os.environ.get('WMA_GPU_MAX_VRAM_GIB')
    if max_vram is not None and max_vram != '':
        try:
            value = float(max_vram)
            if math.isfinite(value) and value > 0:
                config.max_vram_gib = value
        except ValueError:
            pass


def add_gpu_arguments(parser):
    """Add the standard GPU options to an argparse parser used by bin/
    scripts. Apply them with apply_gpu_arguments(args) after parsing."""

    parser.add_argument(
        '-gpu', action='store', dest='gpuMode', default='auto',
        choices=['auto', 'on', 'off'],
        help='GPU acceleration for fiber distance computations: auto '
             '(default, use GPU when available), on, off. Results match '
             'the CPU implementation.')
    parser.add_argument(
        '-gpu_float32', action='store_true', dest='gpuFloat32',
        help='(Fast mode.) Compute GPU distances in float32 instead of '
             'float64. Faster, but distances deviate from the CPU '
             'float64 results by about 1e-6 relative precision, which '
             'can flip cluster labels of near-tie fibers. Default: off.')
    parser.add_argument(
        '-gpu_devices', action='store', dest='gpuDevices', default=None,
        help='Comma separated CUDA device ids to use, e.g. "0,1". '
             'Default: all visible devices. Can also be set with the '
             'WMA_GPU_DEVICES environment variable.')


def apply_gpu_arguments(args):
    """Apply the options created by add_gpu_arguments to the global config."""

    set_gpu(getattr(args, 'gpuMode', 'auto'))
    set_float32(getattr(args, 'gpuFloat32', False))
    devices = getattr(args, 'gpuDevices', None)
    if devices:
        set_devices(devices)
    if config.requested is True:
        use_gpu()


_from_env()
