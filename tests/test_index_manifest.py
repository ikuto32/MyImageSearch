from contextlib import closing
import pathlib
import sqlite3
import tempfile
import unittest
import types
from unittest.mock import Mock, patch

import faiss
import numpy as np

import create_index
from app.domain.domain_object import ImageId, ModelId
from app.infrastructure.local_accessor import LocalAccessor, QwenEmbeddingBackend
from app.infrastructure.model_metadata import (
    INDEX_MANIFEST_SUFFIX,
    SearchModelMetadata,
    file_sha256,
    open_readonly_database,
    read_index_manifest,
    write_model_metadata,
)


class IndexManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.index_dir = self.root / "Test-model"
        self.index_dir.mkdir()
        self.index_path = self.index_dir / "metafiles.index"
        self.db_path = self.index_dir / "sqlite_image_meta.db"
        self.model_id = ModelId("Test", "model")
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("""CREATE TABLE image_meta (
                image_id TEXT PRIMARY KEY, image_path TEXT, meta BLOB,
                image_tags TEXT, aesthetic_quality REAL, pony_aesthetic_quality REAL,
                style_cluster TEXT, rating TEXT
            )""")
            con.executemany("INSERT INTO image_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                ("z", "z.png", np.array([1, 0], dtype=np.float32).tobytes(), "red", 2, 8, "1", "safe"),
                ("a", "a.png", np.array([0, 1], dtype=np.float32).tobytes(), None, 4, 7, None, None),
                ("c", "c.png", np.array([3, 4], dtype=np.float32).tobytes(), "blue", 6, 9, "2", "safe"),
                ("null", "b.png", None, "unindexed", 1, 1, None, None),
                ("wrong", "d.png", np.array([1], dtype=np.float32).tobytes(), "unindexed", 1, 1, None, None),
                ("malformed", "e.png", b"x", "unindexed", 1, 1, None, None),
            ])
            con.commit()
        self.addCleanup(LocalAccessor.load_index_with_metadata.cache_clear)
        self.addCleanup(LocalAccessor.get_mean_meta_vector.cache_clear)

    def build(self):
        # Test actual streaming generation and serialization without PQ training
        # requirements or any embedding model downloads.
        with patch.object(create_index, "createIndex", return_value=faiss.IndexFlatIP(2)):
            create_index.stream_build_faiss(
                str(self.index_dir), 1, 1, 4, 2, str(self.index_path),
                batch_size=2, train_samples=10,
            )

    def accessor(self):
        return LocalAccessor(self.root)

    def test_generated_manifest_matches_only_added_vectors_and_centered_mean(self):
        self.build()
        manifest = read_index_manifest(self.index_path)
        self.assertEqual((manifest["count"], manifest["dimension"]), (3, 2))
        self.assertEqual(manifest["index_sha256"], file_sha256(self.index_path))
        accessor = self.accessor()
        index, items = accessor.load_index_with_metadata(self.model_id, "original")
        self.assertEqual([item.id.id for item in items], ["a", "c", "z"])
        self.assertEqual(items[0].tags.tags, "")
        expected = np.array([[0, 1], [0.6, 0.8], [1, 0]], dtype=np.float32)
        mean = accessor.get_mean_meta_vector(self.model_id)
        np.testing.assert_allclose(mean, expected.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(index.reconstruct_n(0, 3), expected - mean, atol=1e-7)

    def test_legacy_metadata_uses_the_same_vector_selection_rule(self):
        self.build()
        pathlib.Path(str(self.index_path) + INDEX_MANIFEST_SUFFIX).unlink()
        pathlib.Path(str(self.index_path) + ".meta.json").unlink()
        index, items = self.accessor().load_index_with_metadata(self.model_id, "original")
        self.assertEqual(index.ntotal, 3)
        self.assertEqual([item.id.id for item in items], ["a", "c", "z"])

    def test_manifest_keeps_index_row_order_when_database_order_changes(self):
        self.build()
        # New non-indexed metadata rows must not shift any FAISS position.
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("INSERT INTO image_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                "new", "0.png", np.array([1, 1], dtype=np.float32).tobytes(), "new", 1, 1, None, None
            ))
            con.commit()
        _, items = self.accessor().load_index_with_metadata(self.model_id, "original")
        self.assertEqual([item.id.id for item in items], ["a", "c", "z"])

    def test_changed_image_path_fails_instead_of_mislabeling_index_row(self):
        self.build()
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("UPDATE image_meta SET image_path = 'renamed.png' WHERE image_id = 'a'")
            con.commit()
        with self.assertRaisesRegex(ValueError, "changed image paths"):
            self.accessor().load_index_with_metadata(self.model_id, "original")

    def test_replaced_index_with_same_count_is_rejected_by_manifest_hash(self):
        self.build()
        different = faiss.IndexFlatIP(2)
        different.add(np.zeros((3, 2), dtype=np.float32))
        faiss.write_index(different, str(self.index_path))
        with self.assertRaisesRegex(ValueError, "different builds"):
            self.accessor().load_index_with_metadata(self.model_id, "original")

    def test_new_format_missing_manifest_is_rejected(self):
        self.build()
        pathlib.Path(str(self.index_path) + INDEX_MANIFEST_SUFFIX).unlink()
        with self.assertRaisesRegex(ValueError, "manifest is missing"):
            self.accessor().load_index_with_metadata(self.model_id, "original")

    def test_new_format_missing_or_wrong_mean_does_not_fall_back(self):
        self.build()
        mean_path = pathlib.Path(str(self.index_path) + ".mean.npy")
        mean_path.unlink()
        with self.assertRaisesRegex(ValueError, "Mean vector is missing"):
            self.accessor().get_mean_meta_vector(self.model_id)
        np.save(mean_path, np.zeros(2, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "different builds"):
            self.accessor().get_mean_meta_vector(self.model_id)

    def test_legacy_missing_mean_explicitly_uses_uncentered_compatibility(self):
        self.build()
        for suffix in (INDEX_MANIFEST_SUFFIX, ".meta.json", ".mean.npy"):
            pathlib.Path(str(self.index_path) + suffix).unlink()
        with self.assertLogs("app.infrastructure.local_accessor", level="WARNING") as logs:
            self.assertIsNone(self.accessor().get_mean_meta_vector(self.model_id))
        self.assertIn("非中心化", logs.output[0])

    def test_build_failure_keeps_all_existing_index_files_unchanged(self):
        self.build()
        paths = [self.index_path] + [pathlib.Path(str(self.index_path) + suffix) for suffix in (
            INDEX_MANIFEST_SUFFIX, ".meta.json", ".mean.npy"
        )]
        hashes = [file_sha256(path) for path in paths]
        with patch.object(create_index.faiss, "write_index", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                self.build()
        self.assertEqual(hashes, [file_sha256(path) for path in paths])
        self.assertFalse(list(self.index_dir.glob(".*")))

    def test_metadata_lookup_skips_faiss_and_reads_only_requested_ids(self):
        self.index_path.write_bytes(b"No FAISS read is needed for metadata")
        accessor = self.accessor()
        with patch.object(accessor, "_load_search_index", side_effect=AssertionError("FAISS must not load")):
            items = accessor.load_image_metadata(self.model_id, [ImageId("null"), ImageId("a"), ImageId("missing"), ImageId("a")])
            self.assertEqual({item.id.id for item in items}, {"null", "a"})
            self.assertEqual(accessor.load_image_metadata(self.model_id, []), [])

    def test_older_optional_metadata_columns_are_returned_as_unset(self):
        self.index_path.write_bytes(b"registered")
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("ALTER TABLE image_meta DROP COLUMN style_cluster")
            con.execute("ALTER TABLE image_meta DROP COLUMN pony_aesthetic_quality")
            con.execute("ALTER TABLE image_meta DROP COLUMN rating")
            con.commit()
        items = self.accessor().load_image_metadata(self.model_id, [ImageId("a")])
        self.assertEqual((items[0].rating, items[0].style_cluster), ("", ""))
        self.assertEqual(items[0].aesthetic_quality, 4)

    def test_metadata_lookup_chunks_large_id_lists_and_preserves_readonly_database(self):
        self.index_path.write_bytes(b"registered")
        queries = []

        def connect(path):
            con = open_readonly_database(path)
            con.set_trace_callback(queries.append)
            return con

        before = file_sha256(self.db_path)
        with patch("app.infrastructure.local_accessor.open_readonly_database", side_effect=connect):
            self.accessor().load_image_metadata(self.model_id, [ImageId(str(i)) for i in range(1900)])
        self.assertEqual(len([query for query in queries if " IN (" in query]), 3)
        self.assertEqual(file_sha256(self.db_path), before)

    def test_unregistered_model_is_rejected_before_loading_any_backend(self):
        self.index_path.write_bytes(b"registered")
        accessor = self.accessor()
        with patch("app.infrastructure.local_accessor.QwenEmbeddingBackend") as qwen:
            with self.assertRaisesRegex(ValueError, "not registered"):
                accessor.load_embedding_backend(ModelId("unknown/Qwen-model", ""))
        qwen.assert_not_called()

    def test_path_traversal_cannot_register_an_index_outside_metadata_root(self):
        with tempfile.TemporaryDirectory() as external_dir:
            outside = pathlib.Path(external_dir) / "outside-model"
            outside.mkdir()
            (outside / "metafiles.index").write_bytes(b"outside")
            (outside / "sqlite_image_meta.db").write_bytes(b"outside")
            model_id = ModelId(f"../{pathlib.Path(external_dir).name}/outside", "model")
            with self.assertRaisesRegex(ValueError, "not registered"):
                self.accessor().load_index_with_metadata(model_id, "original")

    def test_manifest_with_missing_row_is_rejected(self):
        self.build()
        with closing(sqlite3.connect(str(self.index_path) + INDEX_MANIFEST_SUFFIX)) as con:
            con.execute("DELETE FROM index_rows WHERE position = 1")
            con.commit()
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            self.accessor().load_index_with_metadata(self.model_id, "original")

    def test_nonfinite_embedding_fails_without_replacing_a_valid_index(self):
        self.build()
        before = file_sha256(self.index_path)
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("UPDATE image_meta SET meta = ? WHERE image_id = 'a'", (
                np.array([float("nan"), 1], dtype=np.float32).tobytes(),
            ))
            con.commit()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.build()
        self.assertEqual(file_sha256(self.index_path), before)

    def test_registered_qwen_backend_uses_the_local_metadata_identifier(self):
        self.index_path.write_bytes(b"registered")
        write_model_metadata(self.index_dir, SearchModelMetadata("Qwen/Qwen3-VL-Embedding-2B", "", "Qwen"))
        with patch("app.infrastructure.local_accessor.QwenEmbeddingBackend") as qwen:
            self.accessor().load_embedding_backend(ModelId("Qwen/Qwen3-VL-Embedding-2B", "openai"))
        qwen.assert_called_once_with("Qwen/Qwen3-VL-Embedding-2B")

    def test_qwen_uses_native_transformers_without_remote_code(self):
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = Mock()
        transformers.Qwen3VLModel = Mock()
        with patch.dict("sys.modules", {"transformers": transformers}):
            QwenEmbeddingBackend("Qwen/test")
        self.assertIs(transformers.AutoProcessor.from_pretrained.call_args.kwargs["trust_remote_code"], False)
        self.assertIs(transformers.Qwen3VLModel.from_pretrained.call_args.kwargs["trust_remote_code"], False)


if __name__ == "__main__":
    unittest.main()
