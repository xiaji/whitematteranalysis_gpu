import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import patch

import whitematteranalysis as wma
from whitematteranalysis import cli

ROOT = Path(__file__).resolve().parents[1]


class PackagingMetadataTests(unittest.TestCase):
    def test_pyproject_exists_and_setup_py_removed(self):
        pyproject = ROOT / 'pyproject.toml'
        self.assertTrue(pyproject.is_file())
        self.assertFalse((ROOT / 'setup.py').exists())

        text = pyproject.read_text(encoding='utf-8')
        self.assertIn('name = "whitematteranalysis"', text)
        self.assertIn('wma = "whitematteranalysis.cli:main"', text)
        self.assertIn('version = { attr = "whitematteranalysis.__version__" }', text)
        self.assertIn('exclude = ["tests*", "bin*", "slicer*", "utilities*"]', text)
        # Runtime deps should not pin outdated matplotlib/pandas ceilings or test packages.
        deps_block = text.split('dependencies = [', 1)[1].split(']', 1)[0]
        self.assertNotIn('pytest', deps_block)
        self.assertNotIn('matplotlib>=3.5.0,<3.8', deps_block)
        self.assertNotIn('pandas<2.1.0', deps_block)
        self.assertIn('threadpoolctl>=3.1,<3.7', deps_block)

    def test_all_script_files_in_pyproject_exist(self):
        text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
        block = text.split('script-files = [', 1)[1].split(']', 1)[0]
        scripts = re.findall(r'"([^"]+)"', block)
        self.assertGreaterEqual(len(scripts), 40)
        self.assertIn('bin/wm_apply_ORG_atlas.py', scripts)
        self.assertIn('bin/wm_harden_transform.py', scripts)
        self.assertIn('bin/harden_transform_with_slicer.py', scripts)
        for rel in scripts:
            self.assertTrue((ROOT / rel).is_file(), f'Missing script-file: {rel}')

    def test_version_exposed_and_lazy_import_preserved(self):
        self.assertRegex(wma.__version__, r'^\d+\.\d+\.\d+')
        self.assertIn('cli', wma.__all__)

    def test_dockerfile_and_dockerignore_build_context_is_installable(self):
        import fnmatch
        import shutil
        import tempfile

        dockerfile = ROOT / 'Dockerfile'
        dockerignore = ROOT / '.dockerignore'
        self.assertTrue(dockerfile.is_file())
        self.assertTrue(dockerignore.is_file())

        df_text = dockerfile.read_text(encoding='utf-8')
        self.assertIn('ARG GPU_BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime', df_text)
        self.assertIn('AS cpu', df_text)
        self.assertIn('AS gpu', df_text)
        # Default target (last stage) must be the GPU image.
        self.assertGreater(df_text.index('AS gpu'), df_text.index('AS cpu'))
        self.assertIn('ENTRYPOINT ["wma"]', df_text)
        self.assertIn('gcc', df_text)
        self.assertIn('libgl1', df_text)
        self.assertIn('libxt6', df_text)
        self.assertIn('WMA_ATLAS_CACHE=/tmp/wma-cache/atlas', df_text)
        self.assertIn('TRITON_CACHE_DIR=/tmp/wma-cache/triton', df_text)

        wf_path = ROOT / '.github' / 'workflows' / 'docker-publish.yml'
        self.assertTrue(wf_path.is_file())
        wf_text = wf_path.read_text(encoding='utf-8')
        self.assertIn('REGISTRY: ghcr.io', wf_text)
        self.assertIn('packages: write', wf_text)
        self.assertIn('variant: cu124', wf_text)
        self.assertIn('variant: cu128', wf_text)
        self.assertIn('variant: cpu', wf_text)
        self.assertIn('pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime', wf_text)
        self.assertIn('pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime', wf_text)
        self.assertIn('pull_request', wf_text)

        patterns = [
            line.strip()
            for line in dockerignore.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.strip().startswith('#')
        ]

        def _is_ignored(rel_posix):
            parts = rel_posix.split('/')
            for pat in patterns:
                if pat.startswith('**/'):
                    sub = pat[3:]
                    if any(fnmatch.fnmatch(p, sub) for p in parts) or fnmatch.fnmatch(rel_posix, sub):
                        return True
                elif '/' in pat:
                    if rel_posix == pat or rel_posix.startswith(pat + '/') or fnmatch.fnmatch(rel_posix, pat):
                        return True
                else:
                    if rel_posix == pat or parts[0] == pat or fnmatch.fnmatch(parts[-1], pat):
                        return True
            return False

        # Simulate Docker build context filtering and verify wheel builds cleanly from it.
        with tempfile.TemporaryDirectory(prefix='wma_docker_ctx_') as tmp:
            ctx = Path(tmp) / 'ctx'
            whl_dir = Path(tmp) / 'whls'
            ctx.mkdir()
            whl_dir.mkdir()
            for dirpath, dirnames, filenames in os.walk(ROOT):
                rel_dir = Path(dirpath).relative_to(ROOT)
                rel_dir_posix = '' if str(rel_dir) == '.' else rel_dir.as_posix()
                kept_dirs = []
                for d in dirnames:
                    rel_child = f'{rel_dir_posix}/{d}' if rel_dir_posix else d
                    if not _is_ignored(rel_child):
                        kept_dirs.append(d)
                        (ctx / rel_child).mkdir(parents=True, exist_ok=True)
                dirnames[:] = kept_dirs
                for fname in filenames:
                    rel_file = f'{rel_dir_posix}/{fname}' if rel_dir_posix else fname
                    if not _is_ignored(rel_file):
                        shutil.copy2(Path(dirpath) / fname, ctx / rel_file)

            self.assertFalse((ctx / 'tests').exists())
            self.assertFalse((ctx / 'slicer').exists())
            self.assertTrue((ctx / 'pyproject.toml').is_file())
            self.assertTrue((ctx / 'License.txt').is_file())
            self.assertTrue((ctx / 'README.md').is_file())

            pyproject_text = (ctx / 'pyproject.toml').read_text(encoding='utf-8')
            block = pyproject_text.split('script-files = [', 1)[1].split(']', 1)[0]
            for rel in re.findall(r'"([^"]+)"', block):
                self.assertTrue((ctx / rel).is_file(), f'.dockerignore excluded required script: {rel}')

            import importlib.util
            if importlib.util.find_spec('setuptools.build_meta') is None:
                self.skipTest('setuptools>=64 is required to build a wheel from the Docker context')

            build_code = (
                "import sys; from setuptools import build_meta; "
                "print(build_meta.build_wheel(sys.argv[1]))"
            )
            proc = subprocess.run(
                [sys.executable, '-c', build_code, str(whl_dir)],
                cwd=str(ctx),
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            whls = list(whl_dir.glob('whitematteranalysis-*.whl'))
            self.assertEqual(len(whls), 1)


class WMACLITests(unittest.TestCase):
    def test_version_and_help(self):
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            rc = cli.main(['--version'])
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue().strip(), f'whitematteranalysis {wma.__version__}')

        buf = io.StringIO()
        with patch('sys.stdout', buf):
            rc = cli.main(['--help'])
        self.assertEqual(rc, 0)
        self.assertIn('doctor', buf.getvalue())
        self.assertIn('org', buf.getvalue())

    def test_doctor_json_and_text_without_torch(self):
        info = cli.collect_doctor_info()
        self.assertEqual(info['package']['version'], wma.__version__)
        self.assertIn('effective_backend', info['runtime'])
        self.assertIn('atlas_cache', info['runtime'])
        report = cli.format_doctor_report(info)
        self.assertIn('WhiteMatterAnalysis', report)
        self.assertIn('Core dependencies:', report)
        self.assertIn('Acceleration & optional backends:', report)

        # Verify JSON output and clean execution when torch is blocked.
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT)
        code = (
            "import sys; sys.modules['torch'] = None; "
            "from whitematteranalysis.cli import main; "
            "raise SystemExit(main(['doctor', '--json']))"
        )
        proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertFalse(payload['optional']['torch']['installed'])
        self.assertEqual(payload['package']['version'], wma.__version__)

    def test_doctor_respects_disabled_atlas_cache(self):
        with patch.dict(os.environ, {'WMA_ATLAS_CACHE': 'off'}, clear=False):
            info = cli.collect_doctor_info()
        self.assertFalse(info['runtime']['atlas_cache']['enabled'])
        self.assertIn('disabled', cli.format_doctor_report(info))

    def test_resolve_script_and_forward_args(self):
        for cmd, (script_name, _) in cli.COMMAND_SCRIPTS.items():
            resolved = cli.resolve_script(script_name)
            self.assertIsNotNone(resolved, f'Could not resolve {script_name} for wma {cmd}')
            self.assertTrue(Path(resolved).is_file())

        fake_result = subprocess.CompletedProcess(args=[], returncode=0)
        with patch('whitematteranalysis.cli.subprocess.run', return_value=fake_result) as run_mock:
            rc = cli.main(['org', '--help'])
        self.assertEqual(rc, 0)
        called_cmd = run_mock.call_args[0][0]
        self.assertEqual(called_cmd[0], sys.executable)
        self.assertTrue(called_cmd[1].endswith('wm_apply_ORG_atlas.py'))
        self.assertEqual(called_cmd[2:], ['--help'])

    def test_resolve_script_preserves_unactivated_venv_symlink_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix='wma_venv_resolve_') as tmp:
            root = Path(tmp)
            real_bin = root / 'real_python' / 'bin'
            venv_bin = root / 'venv' / 'bin'
            real_bin.mkdir(parents=True)
            venv_bin.mkdir(parents=True)
            real_py = real_bin / 'python3'
            real_py.write_text('#!/bin/sh\n', encoding='utf-8')
            venv_py = venv_bin / 'python3'
            venv_py.symlink_to(real_py)
            fake_script = venv_bin / 'wm_unique_venv_test_script.py'
            fake_script.write_text('#!/usr/bin/env python\n', encoding='utf-8')

            with patch.object(cli.sysconfig, 'get_path', return_value=str(real_bin)), \
                    patch.object(cli.sys, 'executable', str(venv_py)), \
                    patch.object(cli.sys, 'argv', ['-m']), \
                    patch.object(cli.shutil, 'which', return_value=None):
                resolved = cli.resolve_script('wm_unique_venv_test_script.py')
            self.assertEqual(resolved, str(fake_script))

            # Also verify sysconfig scripts dir takes priority when present.
            with patch.object(cli.sysconfig, 'get_path', return_value=str(venv_bin)), \
                    patch.object(cli.sys, 'executable', str(real_py)), \
                    patch.object(cli.sys, 'argv', ['-m']), \
                    patch.object(cli.shutil, 'which', return_value=None):
                resolved = cli.resolve_script('wm_unique_venv_test_script.py')
            self.assertEqual(resolved, str(fake_script))

    def test_resolve_script_prefers_imported_checkout_over_stale_installed_scripts(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix='wma_stale_scripts_') as tmp:
            stale_bin = Path(tmp) / 'bin'
            stale_bin.mkdir(parents=True)
            stale_harden = stale_bin / 'wm_harden_transform.py'
            stale_harden.write_text('# stale installed script\n', encoding='utf-8')
            stale_py = stale_bin / 'python3'
            stale_py.write_text('#!/bin/sh\n', encoding='utf-8')

            with patch.object(cli.sysconfig, 'get_path', return_value=str(stale_bin)), \
                    patch.object(cli.sys, 'executable', str(stale_py)), \
                    patch.object(cli.sys, 'argv', ['-m']), \
                    patch.object(cli.shutil, 'which', return_value=str(stale_harden)):
                resolved = cli.resolve_script('wm_harden_transform.py')
            self.assertEqual(Path(resolved).resolve(), (ROOT / 'bin' / 'wm_harden_transform.py').resolve())

    def test_unknown_command_returns_exit_2(self):
        buf = io.StringIO()
        with patch('sys.stderr', buf):
            rc = cli.main(['nonexistent-cmd'])
        self.assertEqual(rc, 2)
        self.assertIn("unknown command 'nonexistent-cmd'", buf.getvalue())

    def test_list_json_and_agent_docs_present(self):
        self.assertTrue((ROOT / 'AGENTS.md').is_file())
        self.assertTrue((ROOT / 'llms.txt').is_file())
        self.assertFalse((ROOT / 'slicer' / 'CMakeLists.txt').exists())
        self.assertFalse((ROOT / 'slicer' / 'WhiteMatterAnalysisGPU' / 'WhiteMatterAnalysisGPU.py').exists())

        buf = io.StringIO()
        with patch('sys.stdout', buf):
            rc = cli.main(['list', '--json'])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload['version'], wma.__version__)
        cmd_names = {item['command'] for item in payload['commands']}
        self.assertIn('org', cmd_names)
        self.assertIn('harden', cmd_names)
        self.assertTrue(all(item['available'] for item in payload['commands']))

    def test_masked_pinv_reciprocal_preserves_float32_dtype(self):
        import numpy as np
        e_val_a = np.array([0.0, 2.0, -4.0], dtype=np.float32)
        e_vec_a = np.eye(3, dtype=np.float32)
        max_abs_ev = np.max(np.abs(e_val_a))
        nonzero = np.abs(e_val_a) > (1e-15 * max_abs_ev)
        inv_e_val = np.zeros_like(e_val_a)
        np.divide(1.0, e_val_a, out=inv_e_val, where=nonzero)
        pinv_a = (e_vec_a * inv_e_val[None, :]) @ e_vec_a.T
        self.assertEqual(inv_e_val.dtype, np.float32)
        self.assertEqual(pinv_a.dtype, np.float32)
        np.testing.assert_allclose(np.diag(pinv_a), [0.0, 0.5, -0.25], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
