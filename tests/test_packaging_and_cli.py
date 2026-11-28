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

    def test_unknown_command_returns_exit_2(self):
        buf = io.StringIO()
        with patch('sys.stderr', buf):
            rc = cli.main(['nonexistent-cmd'])
        self.assertEqual(rc, 2)
        self.assertIn("unknown command 'nonexistent-cmd'", buf.getvalue())


if __name__ == '__main__':
    unittest.main()
