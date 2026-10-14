"""Index files written by the builder and their validation when the app loads them."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import faiss
import numpy as np

from app.infrastructure import search_index_loader
from app.infrastructure.index_files import IndexFiles, file_sha256, read_index_manifest
from app.infrastructure.search_index_loader import load_search_index
from helpers import create_catalog
from indexer import faiss_index

COLUMNS = (
    "image_id", "image_path", "meta", "image_tags", "aesthetic_quality",
    "pony_aesthetic_quality", "style_cluster", "rating",
)


def vector(*values) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


class SearchIndexLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # FAISS cannot open paths with non-ASCII characters by name on Windows.
        self.files = create_catalog(Path(temporary.name) / "メタデータ", "Test-model", [
            ("z", "z.png", vector(1, 0), "red", 2, 8, "1", "safe"),
            ("a", "a.png", vector(0, 1), None, 4, 7, None, None),
            ("c", "c.png", vector(3, 4), "blue", 6, None, "2", "general"),
            ("null", "b.png", None, "unindexed", 1, 1, None, None),
            ("wrong", "d.png", vector(1), "unindexed", 1, 1, None, None),
            ("malformed", "e.png", b"x", "unindexed", 1, 1, None, None),
        ], COLUMNS)

    def build(self):
        # Test actual streaming generation and serialization without PQ training
        # requirements or any embedding model downloads.
        with patch.object(faiss_index, "create_ivfpq_index", return_value=faiss.IndexFlatIP(2)):
            faiss_index.build_index(self.files, 1, 1, 4, 2, batch_size=2, train_samples=10)

    def execute(self, path: Path, statement: str, parameters=()):
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(statement, parameters)
            connection.commit()

    def add_image(self):
        self.execute(self.files.database, "INSERT INTO image_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            "new", "n.png", vector(1, 1), "new", 1, 1, None, None,
        ))

    def test_generated_index_holds_only_valid_vectors_centered_in_path_order(self):
        self.build()
        manifest = read_index_manifest(self.files.manifest)
        self.assertEqual((manifest.count, manifest.dimension), (3, 2))
        self.assertEqual(manifest.index_sha256, file_sha256(self.files.index))
        self.assertEqual(manifest.mean_sha256, file_sha256(self.files.mean))

        index = load_search_index(self.files)
        self.assertEqual(list(index.ids), ["a", "c", "z"])
        self.assertEqual(list(index.names), ["a.png", "c.png", "z.png"])
        self.assertTrue(index.names_sorted)
        self.assertEqual((list(index.tags), list(index.ratings)), (["", "blue", "red"], ["", "general", "safe"]))
        self.assertEqual(list(index.style_clusters), ["", "2", "1"])
        np.testing.assert_array_equal(index.aesthetic_scores["original"], [4, 6, 2])
        np.testing.assert_array_equal(index.aesthetic_scores["pony"], [7, np.nan, 8])
        expected = np.array([[0, 1], [0.6, 0.8], [1, 0]], dtype=np.float32)
        np.testing.assert_allclose(index.mean_vector, expected.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(index.index.reconstruct_n(0, 3), expected - index.mean_vector, atol=1e-7)
        self.assertEqual((index.dimension, len(index)), (2, 3))

    def test_index_without_manifest_uses_the_same_vector_selection_rule(self):
        self.build()
        self.files.manifest.unlink()
        self.files.sidecar.unlink()
        index = load_search_index(self.files)
        self.assertEqual(list(index.ids), ["a", "c", "z"])
        self.assertIsNotNone(index.mean_vector)

    def test_uncentered_legacy_index_loads_without_a_mean_vector(self):
        self.build()
        for path in (self.files.manifest, self.files.sidecar, self.files.mean):
            path.unlink()
        with self.assertLogs("app.infrastructure.search_index_loader", level="WARNING") as logs:
            index = load_search_index(self.files)
        self.assertIsNone(index.mean_vector)
        self.assertIn("非中心化", logs.output[0])

    def test_manifest_keeps_index_row_order_when_database_order_changes(self):
        self.build()
        # New non-indexed metadata rows must not shift any FAISS position.
        self.execute(self.files.database, "INSERT INTO image_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            "new", "0.png", vector(1, 1), "new", 1, 1, None, None,
        ))
        self.assertEqual(list(load_search_index(self.files).ids), ["a", "c", "z"])

    def test_rows_out_of_name_order_are_reported_as_unsorted(self):
        self.build()
        self.execute(self.files.manifest, "UPDATE index_rows SET image_path = 'zz.png' WHERE image_id = 'a'")
        self.execute(self.files.database, "UPDATE image_meta SET image_path = 'zz.png' WHERE image_id = 'a'")
        index = load_search_index(self.files)
        self.assertEqual(list(index.names), ["zz.png", "c.png", "z.png"])
        self.assertFalse(index.names_sorted)

    def test_database_without_optional_columns_reads_them_as_unset(self):
        self.build()
        for column in ("style_cluster", "pony_aesthetic_quality", "rating", "image_tags"):
            self.execute(self.files.database, f"ALTER TABLE image_meta DROP COLUMN {column}")
        index = load_search_index(self.files)
        self.assertEqual((list(index.tags), list(index.ratings), list(index.style_clusters)), ([""] * 3,) * 3)
        self.assertTrue(np.isnan(index.aesthetic_scores["pony"]).all())
        np.testing.assert_array_equal(index.aesthetic_scores["original"], [4, 6, 2])

    def test_inconsistent_index_files_are_rejected(self):
        def changed_path():
            self.execute(self.files.database, "UPDATE image_meta SET image_path = 'renamed.png' WHERE image_id = 'a'")

        def deleted_row():
            self.execute(self.files.database, "DELETE FROM image_meta WHERE image_id = 'a'")

        def replaced_index():
            different = faiss.IndexFlatIP(2)
            different.add(np.zeros((3, 2), dtype=np.float32))
            with self.files.index.open("wb") as stream:
                faiss.write_index(different, faiss.PyCallbackIOWriter(stream.write))

        def replaced_mean():
            np.save(self.files.mean, np.zeros(2, dtype=np.float32))

        def missing_manifest_row():
            self.execute(self.files.manifest, "DELETE FROM index_rows WHERE position = 1")

        def broken_sidecar():
            self.files.manifest.unlink()
            self.files.sidecar.write_text("[]", encoding="utf-8")

        for corrupt, message in (
            (changed_path, "changed image paths"),
            (deleted_row, "changed image paths"),
            (replaced_index, "different builds"),
            (self.files.manifest.unlink, "manifest is missing"),
            (self.files.mean.unlink, "Mean vector is missing"),
            (replaced_mean, "different builds"),
            (missing_manifest_row, "row count mismatch"),
            (broken_sidecar, "Invalid index metadata"),
        ):
            with self.subTest(message=message, corrupt=getattr(corrupt, "__name__", "unlink")):
                for path in self.files.all():
                    if path not in (self.files.database, self.files.model_metadata):
                        path.unlink(missing_ok=True)
                self.execute(self.files.database, "DELETE FROM image_meta WHERE image_id = 'a'")
                self.execute(self.files.database, "INSERT INTO image_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                    "a", "a.png", vector(0, 1), None, 4, 7, None, None,
                ))
                self.build()
                load_search_index(self.files)
                corrupt()
                with self.assertRaisesRegex(ValueError, message):
                    load_search_index(self.files)

    def test_index_published_during_a_load_is_not_mixed_with_the_older_format(self):
        self.build()
        sidecar, mean, manifest = self.files.sidecar, self.files.mean, self.files.manifest
        built = {path: path.read_bytes() for path in (sidecar, mean, manifest)}
        read_index = search_index_loader._read_faiss_index
        # On Windows the builder waits for the reader before replacing the index, so only the sidecar and
        # the mean vector may have been published when the reader looks again.
        for published in ((sidecar, mean), (sidecar, mean, manifest)):
            with self.subTest(published=[path.name for path in published]):
                for path in built:
                    path.unlink(missing_ok=True)

                def read_index_then_publish(files):
                    result = read_index(files)
                    for path in published:
                        path.write_bytes(built[path])
                    return result

                with patch.object(search_index_loader, "_read_faiss_index", side_effect=read_index_then_publish):
                    with self.assertRaisesRegex(ValueError, "rebuilt while it was being loaded"):
                        load_search_index(self.files)
                manifest.write_bytes(built[manifest])
                self.assertEqual(len(load_search_index(self.files)), 3)

    @unittest.skipUnless(sys.platform == "win32", "UNC paths exist only on Windows")
    def test_index_on_a_network_share_loads(self):
        self.build()
        local = self.files.directory.resolve()
        share = Path(f"\\\\localhost\\{local.drive[0]}$" + str(local)[2:])
        if not share.is_dir():
            self.skipTest("the administrative share of this drive is not reachable")
        self.assertEqual(list(load_search_index(IndexFiles(share)).ids), ["a", "c", "z"])

    def test_mean_vector_of_another_dimension_is_rejected(self):
        self.build()
        self.files.manifest.unlink()
        self.files.sidecar.unlink()
        np.save(self.files.mean, np.zeros(3, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "does not match the index"):
            load_search_index(self.files)

    def test_build_failure_keeps_all_existing_index_files_unchanged(self):
        self.build()
        paths = [self.files.index, self.files.manifest, self.files.sidecar, self.files.mean]
        hashes = [file_sha256(path) for path in paths]
        with patch.object(faiss_index.faiss, "write_index", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                self.build()
        self.assertEqual(hashes, [file_sha256(path) for path in paths])
        self.assertFalse(list(self.files.directory.glob(".*")))

    def test_publication_retries_while_a_published_file_is_in_use(self):
        self.build()
        self.add_image()
        replace = Path.replace
        refused = []

        def replace_after_two_refusals(source, target):
            if len(refused) < 2:
                refused.append(target)
                raise PermissionError(13, "The process cannot access the file")
            return replace(source, target)

        with (
            patch.object(faiss_index, "_PUBLISH_RETRY_SEC", 0),
            patch.object(Path, "replace", autospec=True, side_effect=replace_after_two_refusals),
        ):
            self.build()
        self.assertEqual(len(refused), 2)
        self.assertEqual(len(load_search_index(self.files)), 4)
        self.assertFalse(list(self.files.directory.glob(".*")))

    def test_publication_that_cannot_finish_leaves_the_published_index_usable(self):
        self.build()
        self.add_image()
        paths = [self.files.index, self.files.manifest, self.files.sidecar, self.files.mean]
        hashes = [file_sha256(path) for path in paths]
        replace = Path.replace

        def replace_unless_manifest(source, target):
            # The file published last stays in use beyond the deadline.
            if target == self.files.manifest:
                raise PermissionError(13, "The process cannot access the file")
            return replace(source, target)

        with (
            patch.object(faiss_index, "_PUBLISH_TIMEOUT_SEC", 0),
            patch.object(Path, "replace", autospec=True, side_effect=replace_unless_manifest),
        ):
            with self.assertRaises(PermissionError):
                self.build()
        self.assertEqual(hashes, [file_sha256(path) for path in paths])
        self.assertFalse(list(self.files.directory.glob(".*")))
        self.assertEqual(len(load_search_index(self.files)), 3)

    @unittest.skipUnless(sys.platform == "win32", "only Windows refuses to replace a file that is open")
    def test_publication_waits_for_a_reader_that_holds_the_index_open(self):
        self.build()
        self.add_image()
        paths = [self.files.index, self.files.manifest, self.files.sidecar, self.files.mean]
        hashes = [file_sha256(path) for path in paths]
        publish = faiss_index._publish
        publishing = threading.Event()

        def publish_and_tell(*arguments):
            publishing.set()
            return publish(*arguments)

        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            patch.object(faiss_index, "_PUBLISH_RETRY_SEC", 0.01),
            patch.object(faiss_index, "_publish", side_effect=publish_and_tell),
        ):
            with self.files.index.open("rb"):
                rebuild = pool.submit(self.build)
                self.assertTrue(publishing.wait(30))
                time.sleep(0.2)
                self.assertFalse(rebuild.done())
                # Nothing is replaced until every published file can be.
                self.assertEqual(hashes, [file_sha256(path) for path in paths])
            rebuild.result(timeout=30)
        self.assertEqual(len(load_search_index(self.files)), 4)

    def test_nonfinite_embedding_fails_without_replacing_a_valid_index(self):
        self.build()
        before = file_sha256(self.files.index)
        self.execute(self.files.database, "UPDATE image_meta SET meta = ? WHERE image_id = 'a'", (
            vector(float("nan"), 1),
        ))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.build()
        self.assertEqual(file_sha256(self.files.index), before)
        self.assertFalse(list(self.files.directory.glob(".*")))

    def test_database_without_any_valid_vector_builds_nothing(self):
        self.execute(self.files.database, "UPDATE image_meta SET meta = NULL")
        self.files.index.unlink()
        self.build()
        self.assertFalse(self.files.index.exists())
        self.assertFalse(self.files.manifest.exists())


class TrainingSampleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.vectors = np.arange(200, dtype=np.float32).reshape(100, 2)
        self.files = create_catalog(Path(temporary.name), "Test-model", [
            (f"id-{row:03d}", f"{row:03d}.png", self.vectors[row].tobytes()) for row in range(100)
        ], ("image_id", "image_path", "meta"))
        self.connection = sqlite3.connect(self.files.database)
        self.addCleanup(self.connection.close)

    def sample(self, count, batch_size):
        return faiss_index.collect_train_samples(self.connection, 2, count, batch_size, total=100)

    def test_all_rows_are_used_in_path_order_when_they_fit(self):
        for batch_size in (7, 100, 1000):
            np.testing.assert_array_equal(self.sample(100, batch_size), self.vectors)
            np.testing.assert_array_equal(self.sample(500, batch_size), self.vectors)

    def test_larger_tables_are_sampled_reproducibly_from_existing_rows(self):
        first = self.sample(10, 7)
        self.assertEqual(first.shape, (10, 2))
        rows = {tuple(row) for row in self.vectors}
        self.assertTrue(all(tuple(row) in rows for row in first))
        self.assertEqual(len({tuple(row) for row in first}), 10)
        # The sample does not depend on how rows are fetched, and later rows can be chosen.
        np.testing.assert_array_equal(first, self.sample(10, 10))
        np.testing.assert_array_equal(first, self.sample(10, 1000))
        self.assertTrue((first[:, 0] >= 20).any())

    def test_mean_vector_is_the_mean_of_normalized_vectors(self):
        normalized = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        mean = faiss_index.compute_mean_vector(self.connection, 2, batch_size=7)
        np.testing.assert_allclose(mean, normalized.mean(axis=0), rtol=1e-6)
        self.assertEqual(mean.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
