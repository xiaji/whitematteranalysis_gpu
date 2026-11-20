"""Backend selection honors explicit GPU requests."""
import argparse
import importlib
import os
import unittest
from unittest.mock import patch

gpu_config = importlib.import_module('whitematteranalysis.gpu.config')


class GPUConfigTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            name: getattr(gpu_config.config, name)
            for name in ('requested', 'float32', 'devices', 'tile_rows',
                         'max_vram_gib', 'strict', 'allow_mps')
            if hasattr(gpu_config.config, name)
        }

    def tearDown(self):
        for name, value in self.settings.items():
            setattr(gpu_config.config, name, value)

    def test_explicit_requests_fail_clearly_without_cuda(self):
        with patch.object(gpu_config.config, 'cuda_available', return_value=False) as available:
            for request in (True, 'on'):
                with self.subTest(request=request), self.assertRaisesRegex(
                        RuntimeError, 'explicitly requested.*CUDA is unavailable'):
                    gpu_config.use_gpu(request)
            gpu_config.set_gpu('on')
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                gpu_config.use_gpu()
            self.assertEqual(available.call_count, 3)

    def test_auto_falls_back_or_uses_available_cuda(self):
        gpu_config.set_gpu('auto')
        with patch.object(gpu_config.config, 'cuda_available', return_value=False) as available:
            self.assertFalse(gpu_config.use_gpu())
            available.assert_called_once_with()
        with patch.object(gpu_config.config, 'cuda_available', return_value=True):
            self.assertTrue(gpu_config.use_gpu())

    def test_off_does_not_probe_cuda(self):
        with patch.object(gpu_config.config, 'cuda_available', side_effect=AssertionError('probed')) as available:
            self.assertFalse(gpu_config.use_gpu(False))
            gpu_config.set_gpu('off')
            self.assertFalse(gpu_config.use_gpu())
            available.assert_not_called()

    def test_environment_and_cli_modes(self):
        with patch.dict(os.environ, {'WMA_GPU': 'on'}):
            gpu_config._from_env()
        self.assertIs(gpu_config.config.requested, True)

        with patch.dict(os.environ):
            os.environ.pop('WMA_GPU', None)
            gpu_config._from_env()
        self.assertIsNone(gpu_config.config.requested)

        parser = argparse.ArgumentParser()
        gpu_config.add_gpu_arguments(parser)
        self.assertEqual(parser.parse_args([]).gpuMode, 'auto')
        with patch.object(gpu_config.config, 'cuda_available', return_value=True):
            for mode, expected in (('on', True), ('off', False)):
                args = parser.parse_args(['-gpu', mode])
                gpu_config.apply_gpu_arguments(args)
                self.assertIs(gpu_config.config.requested, expected)
        with patch.object(gpu_config.config, 'cuda_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                gpu_config.apply_gpu_arguments(parser.parse_args(['-gpu', 'on']))

    def test_strict_mps_and_nonfinite_max_vram_env(self):
        # 1. WMA_GPU_STRICT and strict_enabled / set_strict
        self.assertFalse(gpu_config.strict_enabled())
        gpu_config.set_strict(True)
        self.assertTrue(gpu_config.strict_enabled())
        self.assertFalse(gpu_config.strict_enabled(False))
        gpu_config.set_strict(False)
        self.assertFalse(gpu_config.strict_enabled())

        with patch.dict(os.environ, {'WMA_GPU_STRICT': '1'}):
            gpu_config._from_env()
        self.assertTrue(gpu_config.strict_enabled())
        with patch.dict(os.environ):
            os.environ.pop('WMA_GPU_STRICT', None)
            gpu_config._from_env()
        self.assertFalse(gpu_config.strict_enabled())

        # 2. WMA_GPU_MAX_VRAM_GIB ignores inf / -inf / nan / <=0
        gpu_config.config.max_vram_gib = None
        for bad in ('inf', '-inf', 'nan', '0', '-2.5'):
            with patch.dict(os.environ, {'WMA_GPU_MAX_VRAM_GIB': bad}):
                gpu_config._from_env()
            self.assertIsNone(gpu_config.config.max_vram_gib, bad)

        # 3. WMA_GPU_MPS=1 activates MPS ONLY when float32=True, cuda=False, mps=True
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required')

        with patch.dict(os.environ, {'WMA_GPU_MPS': '1', 'WMA_GPU_FLOAT32': '0'}):
            gpu_config._from_env()
        self.assertTrue(gpu_config.config.allow_mps)
        gpu_config.set_gpu('auto')

        # Missing float32: must not probe MPS and must stay unavailable / raise on 'on'
        with patch.object(gpu_config.config, 'cuda_available', return_value=False), \
                patch.object(gpu_config.config, 'mps_available', side_effect=AssertionError('probed MPS without float32')):
            self.assertFalse(gpu_config.use_gpu())
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                gpu_config.use_gpu('on')

        # Missing WMA_GPU_MPS: must not probe MPS even when float32=True
        with patch.dict(os.environ, {'WMA_GPU_MPS': '0', 'WMA_GPU_FLOAT32': '1'}):
            gpu_config._from_env()
        with patch.object(gpu_config.config, 'cuda_available', return_value=False), \
                patch.object(gpu_config.config, 'mps_available', side_effect=AssertionError('probed MPS without WMA_GPU_MPS')):
            self.assertFalse(gpu_config.use_gpu())
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                gpu_config.use_gpu('on')

        # Both WMA_GPU_MPS=1 and float32=True, but mps_available=False
        with patch.dict(os.environ, {'WMA_GPU_MPS': '1', 'WMA_GPU_FLOAT32': '1'}):
            gpu_config._from_env()
        with patch.object(gpu_config.config, 'cuda_available', return_value=False), \
                patch.object(gpu_config.config, 'mps_available', return_value=False):
            self.assertFalse(gpu_config.use_gpu())
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                gpu_config.use_gpu('on')

        # All conditions met -> MPS active
        with patch.object(gpu_config.config, 'cuda_available', return_value=False), \
                patch.object(gpu_config.config, 'mps_available', return_value=True):
            self.assertTrue(gpu_config.use_gpu())
            self.assertTrue(gpu_config.use_gpu('on'))
            self.assertEqual(gpu_config.get_device(), torch.device('mps'))
            self.assertEqual(gpu_config.visible_device_count(), 1)
            self.assertFalse(gpu_config.use_gpu('off'))

        # CUDA takes precedence over MPS even when WMA_GPU_MPS=1 and float32=True
        with patch.object(gpu_config.config, 'cuda_available', return_value=True), \
                patch.object(gpu_config.config, 'mps_available', return_value=True):
            self.assertTrue(gpu_config.use_gpu())
            self.assertEqual(gpu_config.get_device(), torch.device('cuda', 0))


if __name__ == '__main__':
    unittest.main()
