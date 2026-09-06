import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen_collate import _install_dummy_modules  # noqa: E402

_install_dummy_modules()

import create_index  # noqa: E402


class SidecarTagsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = Path("nested/photo.jpg")
        self.sidecar = self.root / "nested/photo.jpg.tags.json"
        self.sidecar.parent.mkdir()

    def load(self):
        return create_index.load_sidecar_tags(str(self.root), str(self.image))

    def test_absent_sidecar_leaves_image_unclassified_without_warning(self):
        with mock.patch("builtins.print") as warning:
            self.assertEqual(self.load(), {"rating": "", "tags": []})
        warning.assert_not_called()

    def test_nested_sidecar_preserves_unicode_tags_and_optional_fields(self):
        for data in ({"rating": "general", "tags": ["山", "sky"]}, {"tags": ["snow"]}, {}):
            with self.subTest(data=data):
                self.sidecar.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")
                self.assertEqual(self.load(), {"rating": data.get("rating", ""), "tags": data.get("tags", [])})

    def test_invalid_sidecar_warns_and_does_not_block_image_indexing(self):
        contents = (b"{", b"\xff", b"[]", b'{"rating": 1}', b'{"tags": "sky"}', b'{"tags": [1]}')
        for content in contents:
            with self.subTest(content=content), mock.patch("builtins.print") as warning:
                self.sidecar.write_bytes(content)
                self.assertEqual(self.load(), {"rating": "", "tags": []})
                self.assertIn(str(self.sidecar), warning.call_args.args[0])

    def test_unreadable_sidecar_warns_and_leaves_image_unclassified(self):
        with mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")), mock.patch("builtins.print") as warning:
            self.assertEqual(self.load(), {"rating": "", "tags": []})
        self.assertIn("denied", warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
