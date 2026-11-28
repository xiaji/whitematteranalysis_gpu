"""Unified ``wma`` command-line interface and environment diagnostics."""

from importlib import import_module
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

from . import __version__

COMMAND_SCRIPTS = {
    'org': (
        'wm_apply_ORG_atlas.py',
        'Single-process ORG / multi-atlas parcellation (single subject or batch)',
    ),
    'harden': (
        'wm_harden_transform.py',
        'Apply rigid/affine or BSpline/Composite ITK transforms (pure-Python/VTK or Slicer)',
    ),
    'register': (
        'wm_register_to_atlas_new.py',
        'Register subject tractography to an atlas',
    ),
    'cluster': (
        'wm_cluster_from_atlas.py',
        'Cluster subject tractography using a spectral fiber atlas',
    ),
    'build-atlas': (
        'wm_cluster_atlas.py',
        'Construct a spectral clustering atlas from multiple subjects',
    ),
    'remove-outliers': (
        'wm_cluster_remove_outliers.py',
        'Remove anatomical outlier fibers from clustered tractography',
    ),
    'separate': (
        'wm_separate_clusters_by_hemisphere.py',
        'Separate bilateral fiber clusters into left/right/commissural hemispheres',
    ),
    'download-atlas': (
        'wm_download_anatomically_curated_atlas.py',
        'Download anatomically curated ORG atlases',
    ),
}

_TRACKED_ENV_VARS = (
    'WMA_GPU',
    'WMA_GPU_FLOAT32',
    'WMA_GPU_DEVICES',
    'WMA_GPU_STRICT',
    'WMA_GPU_MPS',
    'WMA_GPU_TILE',
    'WMA_GPU_MIN_PAIR_WORK',
    'WMA_GPU_MAX_VRAM_GIB',
    'WMA_GPU_PROCESS_POOL',
    'WMA_CPU_FAST',
    'WMA_CPU_THREADS',
    'WMA_CPU_BLAS_THREADS',
    'WMA_NONRIGID_ANALYTIC_GRAD',
    'WMA_VTP_COMPRESSION',
    'WMA_IO_PREFETCH',
    'WMA_ATLAS_CACHE',
)


def _package_version(module_name):
    try:
        mod = import_module(module_name)
    except Exception:
        return None
    if module_name == 'vtk':
        try:
            return mod.vtkVersion.GetVTKVersion()
        except Exception:
            pass
    return getattr(mod, '__version__', 'installed')


def _threadpoolctl_blas_cap_supported(version):
    if not version:
        return False
    try:
        parts = tuple(int(p) for p in version.split('.')[:2])
        return (3, 1) <= parts < (3, 7)
    except Exception:
        return False


def _resolve_atlas_cache_info():
    raw = os.environ.get('WMA_ATLAS_CACHE')
    if raw and raw.strip().lower() in ('off', '0', 'false'):
        return {'enabled': False, 'path': None, 'exists': False, 'writable': False}
    if raw:
        root = Path(raw).expanduser()
    else:
        base = os.environ.get('LOCALAPPDATA') if os.name == 'nt' else os.environ.get('XDG_CACHE_HOME')
        root = Path(base or (Path.home() / '.cache')) / 'whitematteranalysis'
    exists = root.exists()
    probe = root if exists else root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    writable = os.access(probe, os.W_OK) if probe.exists() else False
    return {
        'enabled': True,
        'path': str(root),
        'exists': exists,
        'writable': writable,
    }


def collect_doctor_info():
    """Gather runtime, dependency, GPU/Triton, cache, and Slicer diagnostics."""
    pkg_dir = str(Path(__file__).resolve().parent)
    core_packages = {
        name: _package_version(name)
        for name in ('numpy', 'scipy', 'vtk', 'joblib', 'nibabel', 'pandas',
                     'matplotlib', 'statsmodels', 'threadpoolctl', 'xlrd')
    }
    h5py_ver = _package_version('h5py')

    torch_info = {
        'installed': False,
        'version': None,
        'cuda_build': None,
        'cuda_available': False,
        'cuda_device_count': 0,
        'cuda_devices': [],
        'mps_available': False,
    }
    try:
        import torch
        torch_info['installed'] = True
        torch_info['version'] = getattr(torch, '__version__', None)
        torch_info['cuda_build'] = getattr(torch.version, 'cuda', None)
        cuda_ok = bool(torch.cuda.is_available())
        torch_info['cuda_available'] = cuda_ok
        if cuda_ok:
            count = int(torch.cuda.device_count())
            torch_info['cuda_device_count'] = count
            devices = []
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                devices.append({
                    'index': idx,
                    'name': props.name,
                    'capability': f'{props.major}.{props.minor}',
                    'total_memory_gib': round(props.total_memory / (1024 ** 3), 2),
                })
            torch_info['cuda_devices'] = devices
        torch_info['mps_available'] = bool(
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        )
    except Exception:
        pass

    triton_ver = _package_version('triton')
    from . import gpu
    try:
        gpu_active = bool(gpu.use_gpu())
        if gpu_active:
            effective_backend = (
                'cuda' if torch_info['cuda_available'] else (
                    'mps' if torch_info['mps_available'] else 'gpu'
                )
            )
        else:
            effective_backend = 'cpu_fast' if gpu.config.cpu_fast_enabled() else 'cpu_legacy'
    except Exception as exc:
        effective_backend = f'error ({exc})'

    env_values = {key: os.environ[key] for key in _TRACKED_ENV_VARS if key in os.environ}
    slicer_path = shutil.which('Slicer')

    return {
        'package': {
            'name': 'whitematteranalysis',
            'version': __version__,
            'location': pkg_dir,
        },
        'python': {
            'version': platform.python_version(),
            'executable': sys.executable,
            'platform': platform.platform(),
            'machine': platform.machine(),
        },
        'dependencies': core_packages,
        'threadpoolctl_blas_cap': _threadpoolctl_blas_cap_supported(core_packages.get('threadpoolctl')),
        'optional': {
            'h5py': h5py_ver,
            'triton': triton_ver,
            'torch': torch_info,
            'slicer': slicer_path,
        },
        'runtime': {
            'effective_backend': effective_backend,
            'slicer_free_org_supported': core_packages.get('vtk') is not None,
            'atlas_cache': _resolve_atlas_cache_info(),
            'env': env_values,
        },
    }


def format_doctor_report(info):
    """Render a human-readable diagnostics report from ``collect_doctor_info``."""
    pkg = info['package']
    py = info['python']
    deps = info['dependencies']
    opt = info['optional']
    torch_info = opt['torch']
    rt = info['runtime']
    cache = rt['atlas_cache']

    lines = [
        f"WhiteMatterAnalysis {pkg['version']} ({pkg['location']})",
        f"Python: {py['version']} ({py['executable']})",
        f"Platform: {py['platform']} [{py['machine']}]",
        '',
        'Core dependencies:',
    ]
    for name, ver in deps.items():
        status = ver if ver is not None else 'MISSING'
        extra = ''
        if name == 'threadpoolctl' and ver is not None:
            extra = ' (BLAS cap active)' if info['threadpoolctl_blas_cap'] else ' (outside 3.1-<3.7, BLAS cap inactive)'
        lines.append(f'  - {name}: {status}{extra}')

    lines.append('')
    lines.append('Acceleration & optional backends:')
    lines.append(f"  - effective backend: {rt['effective_backend']}")
    if torch_info['installed']:
        lines.append(
            f"  - torch: {torch_info['version']} (cuda_build={torch_info['cuda_build']}, "
            f"cuda_available={torch_info['cuda_available']}, mps_available={torch_info['mps_available']})"
        )
        for dev in torch_info['cuda_devices']:
            lines.append(
                f"      GPU {dev['index']}: {dev['name']} "
                f"(sm_{dev['capability'].replace('.', '')}, {dev['total_memory_gib']} GiB)"
            )
    else:
        lines.append('  - torch: not installed (CPU-only mode)')
    lines.append(f"  - triton: {opt['triton'] or 'not installed (PyTorch CUDA fallback)'}")
    lines.append(f"  - h5py (ITK .h5 transforms): {opt['h5py'] or 'not installed'}")
    lines.append(
        f"  - 3D Slicer: {opt['slicer'] or 'not found (not required for tensor-free rigid/nonrigid ORG)'}"
    )

    lines.append('')
    if cache['enabled']:
        lines.append(
            f"Atlas cache: {cache['path']} (exists={cache['exists']}, writable={cache['writable']})"
        )
    else:
        lines.append('Atlas cache: disabled (WMA_ATLAS_CACHE=off)')

    if rt['env']:
        lines.append('WMA environment variables:')
        for key, val in sorted(rt['env'].items()):
            lines.append(f'  - {key}={val}')
    else:
        lines.append('WMA environment variables: (none set; using defaults)')

    return '\n'.join(lines)


def resolve_script(script_name):
    """Locate an installed or source-tree WMA script by filename."""
    candidates = []
    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / 'pyproject.toml').is_file():
        candidates.append(repo_root / 'bin' / script_name)
        candidates.append(repo_root / 'utilities' / script_name)
    scripts_dir = sysconfig.get_path('scripts')
    if scripts_dir:
        candidates.append(Path(scripts_dir) / script_name)
    if sys.executable:
        exe_path = Path(sys.executable)
        candidates.append(exe_path.parent / script_name)
    if sys.argv and sys.argv[0]:
        argv_path = Path(sys.argv[0])
        candidates.append(argv_path.parent / script_name)
        candidates.append(argv_path.resolve().parent / script_name)
    if sys.executable:
        candidates.append(Path(sys.executable).resolve().parent / script_name)
    candidates.append(repo_root / 'bin' / script_name)
    candidates.append(repo_root / 'utilities' / script_name)

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return key
    found = shutil.which(script_name)
    if found:
        return found
    return None


def _print_help():
    lines = [
        f'usage: wma [-h] [--version] <command> [args ...]',
        '',
        'WhiteMatterAnalysis unified CLI (all legacy wm_*.py scripts remain available).',
        '',
        'commands:',
        '  doctor           Inspect Python, VTK, CUDA/PyTorch, Triton, cache, and Slicer status',
        '  list             List wma subcommands and available script paths',
    ]
    for cmd, (script_name, desc) in COMMAND_SCRIPTS.items():
        lines.append(f'  {cmd:<16} {desc} [{script_name}]')
    lines.append('')
    lines.append("Run 'wma <command> --help' for command-specific options.")
    print('\n'.join(lines))


def _print_list():
    print(f'WhiteMatterAnalysis {__version__} commands:')
    for cmd, (script_name, desc) in COMMAND_SCRIPTS.items():
        resolved = resolve_script(script_name) or 'not found'
        print(f'  wma {cmd:<15} -> {script_name} ({resolved})')
        print(f'      {desc}')


def main(argv=None):
    """Entry point for the ``wma`` console script."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h', '--help', 'help'):
        _print_help()
        return 0
    if args[0] in ('-V', '--version'):
        print(f'whitematteranalysis {__version__}')
        return 0

    cmd, rest = args[0], args[1:]
    if cmd == 'doctor':
        if any(flag in ('-h', '--help') for flag in rest):
            print('usage: wma doctor [--json]')
            return 0
        as_json = '--json' in rest
        unknown = [a for a in rest if a != '--json']
        if unknown:
            print(f"wma doctor: unrecognized arguments: {' '.join(unknown)}", file=sys.stderr)
            return 2
        info = collect_doctor_info()
        if as_json:
            print(json.dumps(info, indent=2, sort_keys=True))
        else:
            print(format_doctor_report(info))
        return 0

    if cmd == 'list':
        _print_list()
        return 0

    if cmd in COMMAND_SCRIPTS:
        script_name, _ = COMMAND_SCRIPTS[cmd]
        script_path = resolve_script(script_name)
        if script_path is None:
            print(f'wma {cmd}: could not locate script {script_name}', file=sys.stderr)
            return 1
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[1]
        if (repo_root / 'pyproject.toml').is_file():
            root_str = str(repo_root)
            existing_pp = env.get('PYTHONPATH', '')
            if root_str not in existing_pp.split(os.pathsep):
                env['PYTHONPATH'] = f'{root_str}{os.pathsep}{existing_pp}' if existing_pp else root_str
        completed = subprocess.run([sys.executable, script_path, *rest], env=env)
        return int(completed.returncode)

    print(f"wma: unknown command '{cmd}'. Run 'wma --help' to see available commands.", file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
