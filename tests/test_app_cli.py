import importlib.util
import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("app_entry", ROOT / "app.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class AppCliTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            'MYIMAGESEARCH_IMAGE_DIR': '',
            'MYIMAGESEARCH_META_DIR': '',
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_defaults_resolve_paths_from_project(self):
        args = entry.parse_args([])
        self.assertEqual((args.image_dir, args.meta_dir), (ROOT / 'images', ROOT / 'clip_meta'))
        self.assertEqual((args.port, args.skip_warmup), (80, False))

    def test_environment_overrides_project_defaults(self):
        os.environ['MYIMAGESEARCH_IMAGE_DIR'] = 'external-images'
        args = entry.parse_args([])
        self.assertEqual((args.image_dir, args.meta_dir), (Path('external-images'), ROOT / 'clip_meta'))
        os.environ['MYIMAGESEARCH_META_DIR'] = 'external-metadata'
        args = entry.parse_args([])
        self.assertEqual((args.image_dir, args.meta_dir), (Path('external-images'), Path('external-metadata')))

    def test_warmup_is_explicit(self):
        self.assertFalse(entry.parse_args(['--warmup']).skip_warmup)
        self.assertTrue(entry.parse_args(['--skip-warmup']).skip_warmup)

    def test_host_can_be_limited_to_this_pc(self):
        args = entry.parse_args(['--host', '127.0.0.1'])
        self.assertEqual(args.host, '127.0.0.1')

    def test_local_mode_overrides_environment(self):
        os.environ['MYIMAGESEARCH_IMAGE_DIR'] = 'external-images'
        os.environ['MYIMAGESEARCH_META_DIR'] = 'external-metadata'
        args = entry.parse_args(['--local', '--skip-warmup', '--port', '5000'])
        self.assertEqual((args.image_dir, args.meta_dir), (ROOT / 'images', ROOT / 'clip_meta'))
        self.assertEqual((args.port, args.skip_warmup), (5000, True))

    def test_explicit_paths_override_environment_and_local_mode(self):
        os.environ['MYIMAGESEARCH_IMAGE_DIR'] = 'external-images'
        os.environ['MYIMAGESEARCH_META_DIR'] = 'external-metadata'
        for profile in ([], ['--local']):
            with self.subTest(profile=profile):
                args = entry.parse_args(profile + ['--image-dir', 'custom', '--meta-dir', 'metadata', '--host', 'localhost'])
                self.assertEqual((args.image_dir, args.meta_dir, args.host), (Path('custom'), Path('metadata'), 'localhost'))
        args = entry.parse_args(['--image-dir', 'custom'])
        self.assertEqual((args.image_dir, args.meta_dir), (Path('custom'), Path('external-metadata')))

    def test_invalid_port_rejected(self):
        with self.assertRaises(SystemExit):
            entry.parse_args(['--port', '0'])

    def test_startup_model_uses_available_model_when_default_is_missing(self):
        other = SimpleNamespace(id=SimpleNamespace(model_name='Qwen/example', pretrained='local'))
        preferred = SimpleNamespace(id=SimpleNamespace(model_name='ViT-L-14', pretrained='openai'))
        self.assertIs(entry.select_startup_model([other]), other.id)
        self.assertIs(entry.select_startup_model([other, preferred]), preferred.id)
        with self.assertRaises(ValueError):
            entry.select_startup_model([])
