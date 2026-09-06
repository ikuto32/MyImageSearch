import pathlib
from contextlib import closing
import sqlite3
import tempfile
import threading
import unittest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from app.domain.domain_object import ImageId, ModelId
from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.model_metadata import SearchModelMetadata, write_model_metadata


class LocalAccessorStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = pathlib.Path(self.temp_dir.name)
        self.accessor = LocalAccessor(self.root)
        self.addCleanup(LocalAccessor.load_index_with_metadata.cache_clear)
        self.addCleanup(LocalAccessor.load_startup_image_items.cache_clear)

    def create_index(self, directory):
        index_dir = self.root / directory
        index_dir.mkdir()
        (index_dir / "metafiles.index").write_bytes(b"fake index read by mock")
        with closing(sqlite3.connect(index_dir / "sqlite_image_meta.db")) as con:
            con.execute("""CREATE TABLE image_meta (
                image_id TEXT, image_path TEXT, image_tags TEXT,
                aesthetic_quality REAL, pony_aesthetic_quality REAL,
                style_cluster TEXT, rating TEXT, meta BLOB
            )""")
            con.executemany("INSERT INTO image_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                ("b", "b.png", "blue", 2.0, 8.0, "2", "safe", b"\0" * 8),
                ("a", "a.png", "red", 1.0, 9.0, "1", "safe", b"\0" * 8),
            ])
            con.commit()
        return index_dir

    def test_aesthetic_models_share_index_but_keep_their_scores_and_row_order(self):
        index_dir = self.create_index("ViT-L-14-openai")
        model_id = ModelId("ViT-L-14", "openai")
        with patch("app.infrastructure.local_accessor.faiss.read_index", return_value=SimpleNamespace(d=2, ntotal=2)) as read:
            original_index, original = self.accessor.load_index_with_metadata(model_id, "original")
            pony_index, pony = self.accessor.load_index_with_metadata(model_id, "pony")
            self.accessor.load_index_with_metadata(model_id, "original")

        self.assertIs(original_index, pony_index)
        read.assert_called_once()
        self.assertEqual([item.id.id for item in original], ["a", "b"])
        self.assertEqual([item.id.id for item in pony], ["a", "b"])
        self.assertEqual([item.aesthetic_quality for item in original], [1.0, 2.0])
        self.assertEqual([item.aesthetic_quality for item in pony], [9.0, 8.0])

    def test_concurrent_requests_read_the_same_index_once(self):
        index_dir = self.create_index("ViT-L-14-openai")
        barrier = threading.Barrier(4)

        def load_index(_):
            barrier.wait(timeout=5)
            return self.accessor._load_search_index(index_dir / "metafiles.index")

        with patch("app.infrastructure.local_accessor.faiss.read_index", return_value=object()) as read:
            with ThreadPoolExecutor(max_workers=4) as executor:
                indexes = list(executor.map(load_index, range(4)))

        read.assert_called_once()
        self.assertTrue(all(index is indexes[0] for index in indexes))

    def test_failed_index_load_can_be_retried(self):
        index_path = self.root / "metafiles.index"
        index_path.write_bytes(b"fake index")
        expected = object()
        with patch("app.infrastructure.local_accessor.faiss.read_index", side_effect=[OSError("unavailable"), expected]):
            with self.assertRaises(OSError):
                self.accessor._load_search_index(index_path)
            self.assertIs(self.accessor._load_search_index(index_path), expected)

    def test_startup_resolves_qwen_safe_and_metadata_named_directories(self):
        for dirname in ["Qwen--Qwen3-VL-Embedding-2B", "custom-index"]:
            with self.subTest(dirname=dirname):
                with tempfile.TemporaryDirectory(dir=self.root) as case_dir:
                    index_dir = self.create_index(pathlib.Path(case_dir) / dirname)
                    write_model_metadata(index_dir, SearchModelMetadata(
                        "Qwen/Qwen3-VL-Embedding-2B", "", "Qwen embedding"
                    ))
                    items, paths = LocalAccessor(case_dir).load_startup_image_items(
                        ModelId("Qwen/Qwen3-VL-Embedding-2B", "openai")
                    )
                self.assertEqual([item.id.id for item in items], ["a", "b"])
                self.assertEqual(paths, {ImageId("a"): pathlib.Path("a.png"), ImageId("b"): pathlib.Path("b.png")})


if __name__ == "__main__":
    unittest.main()
