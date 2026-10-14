import contextlib
from contextlib import closing, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

from app.domain.domain_object import ModelId
from app.infrastructure.catalog_database import CatalogDatabase
from app.infrastructure.model_registry import ModelRegistry
from indexer import database, image_scan
from indexer.database import (
    collect_orphan_db_records,
    connect_db,
    delete_orphan_db_records,
    finalize_db,
    find_images_without_embedding,
    image_id_for_path,
    infer_meta_dimension,
    init_db,
    upsert_image_meta,
)
from indexer.image_scan import ImageScan, scan_image_dir


def row(image_id, path, meta=None, rating="general", tags="", blake3=None):
    return (image_id, path, meta, None, None, "", rating, tags, "2026-01-01T00:00:00+00:00", "search", None, blake3)


def vector(dimension) -> bytes:
    return np.zeros(dimension, dtype=np.float32).tobytes()


class IndexerDatabaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "Test-model"
        self.directory.mkdir()
        self.connection = connect_db(self.directory)
        self.addCleanup(self.connection.close)
        init_db(self.connection)
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        progress = mock.patch.object(database.tqdm, "tqdm", side_effect=lambda iterable, **_: iterable)
        progress.start()
        self.addCleanup(progress.stop)

    def insert(self, *rows):
        upsert_image_meta(self.connection, rows)
        self.connection.commit()

    def test_initialization_is_repeatable_and_upgrades_older_tables(self):
        init_db(self.connection)
        self.connection.close()
        older = self.root / "Older-model"
        older.mkdir()
        with closing(sqlite3.connect(older / "sqlite_image_meta.db")) as connection:
            # Column names differ only in case from the current ones.
            connection.execute(
                "CREATE TABLE image_meta (image_id text PRIMARY KEY, image_path text, meta blob, IMAGE_TAGS text)"
            )
            connection.execute("INSERT INTO image_meta VALUES ('a', 'a.png', NULL, 'old')")
            connection.commit()
        with closing(connect_db(older)) as upgraded:
            init_db(upgraded)
            init_db(upgraded)
            columns = {info[1].lower(): info[2].lower() for info in upgraded.execute("PRAGMA table_info(image_meta)")}
            self.assertEqual(set(columns), {name.lower() for name in database._COLUMNS})
            self.assertEqual((columns["rating"], columns["pony_aesthetic_quality"]), ("text", "real"))
            # Everything the builder does afterwards works on the upgraded table.
            upsert_image_meta(upgraded, [row("b", "b.png", vector(2), "general", "new")])
            finalize_db(upgraded)
            self.assertEqual(
                upgraded.execute("SELECT image_id, rating, image_tags FROM image_meta ORDER BY image_id").fetchall(),
                [("a", None, "old"), ("b", "general", "new")],
            )

    def test_upsert_replaces_everything_but_the_path_of_an_existing_image(self):
        self.insert(row("a", "a.png", vector(2), "general", "old", blake3="11" * 32))
        self.insert(row("a", "ignored.png", vector(4), "explicit", "new", blake3="22" * 32))
        stored = self.connection.execute(
            "SELECT image_path, length(meta), rating, image_tags, image_blake3 FROM image_meta"
        ).fetchall()
        self.assertEqual(stored, [("a.png", 16, "explicit", "new", "22" * 32)])

    def test_images_without_a_matching_embedding_are_found_without_reading_blobs(self):
        self.insert(
            row("ok", "ok.png", vector(4)), row("other-dimension", "o.png", vector(2)),
            row("no-meta", "n.png", None), row("not-requested", "x.png", None),
        )
        requested = ["ok", "other-dimension", "no-meta", "new"]
        queries = []
        self.connection.set_trace_callback(queries.append)
        self.assertEqual(
            find_images_without_embedding(self.connection, requested, 4), ["other-dimension", "no-meta", "new"],
        )
        self.connection.set_trace_callback(None)
        self.assertFalse(any("image_meta.meta," in query or "SELECT meta" in query for query in queries), queries)
        self.assertEqual(find_images_without_embedding(self.connection, requested, None), ["no-meta", "new"])
        self.assertEqual(find_images_without_embedding(self.connection, [], 4), [])
        temporary_tables = self.connection.execute("SELECT name FROM sqlite_temp_master").fetchall()
        self.assertEqual(temporary_tables, [])

    def test_many_ids_are_checked_in_their_given_order(self):
        image_ids = [f"id-{index:05d}" for index in range(25_000)]
        self.insert(*(row(image_id, f"{image_id}.png", vector(2)) for image_id in image_ids[::2]))
        self.assertEqual(find_images_without_embedding(self.connection, image_ids, 2), image_ids[1::2])

    def test_orphan_records_are_reported_and_deleted_only_on_request(self):
        self.insert(row("kept", "kept.png", vector(2)), row("gone", "gone.png", vector(2)), row("unnamed", None))
        orphans = collect_orphan_db_records(self.connection, ["kept", "new"])
        self.assertEqual(orphans.total_db_count, 3)
        self.assertEqual(sorted(orphans.orphan_records, key=str), [("gone", "gone.png"), ("unnamed", None)])
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM image_meta").fetchone()[0], 3)
        delete_orphan_db_records(self.connection, orphans)
        self.assertEqual(self.connection.execute("SELECT image_id FROM image_meta").fetchall(), [("kept",)])
        self.assertEqual(collect_orphan_db_records(self.connection, ["kept"]).orphan_count, 0)
        self.assertEqual(collect_orphan_db_records(self.connection, []).orphan_count, 1)

    def test_embedding_dimension_is_inferred_from_stored_vectors(self):
        self.assertIsNone(infer_meta_dimension(self.connection))
        self.insert(row("empty", "e.png", None))
        self.assertIsNone(infer_meta_dimension(self.connection))
        self.insert(row("a", "a.png", vector(768)), row("b", "b.png", vector(768)))
        self.assertEqual(infer_meta_dimension(self.connection), 768)
        # A record left from before the dimension was changed must not decide the dimension silently.
        self.insert(row("0-deleted", "0.png", vector(4)))
        with self.assertRaisesRegex(ValueError, r"4 \(1 rows\), 768 \(2 rows\).*--search-model-out-dim"):
            infer_meta_dimension(self.connection)
        self.connection.execute("DELETE FROM image_meta WHERE image_id IN ('0-deleted', 'b')")
        self.connection.execute("UPDATE image_meta SET meta = x'000102' WHERE image_id = 'a'")
        with self.assertRaisesRegex(ValueError, "Invalid meta byte size"):
            infer_meta_dimension(self.connection)

    def test_embedding_dimension_is_found_without_loading_the_embeddings(self):
        self.insert(row("a", "a.png", vector(4)), row("b", "b.png", vector(4)))
        statements = []
        self.connection.set_trace_callback(statements.append)
        self.assertEqual(infer_meta_dimension(self.connection), 4)
        self.connection.set_trace_callback(None)
        self.assertTrue(statements)
        meta = [info[1] for info in self.connection.execute("PRAGMA table_info(image_meta)")].index("meta")
        for statement in statements:
            for _address, opcode, _cursor, column, _p3, _p4, flags, _comment in self.connection.execute(
                "EXPLAIN " + statement
            ):
                if opcode == "Column" and column == meta:
                    # 0x40 and 0x80 mark a read of only the length or the type of a value. Without them
                    # SQLite loads each embedding, and an aggregating query copies them all into its sorter.
                    self.assertTrue(int(flags) & 0xC0, statement)

    def test_finalized_database_serves_browsing_and_category_counts_from_indexes(self):
        self.connection.execute("CREATE INDEX idx_image_meta_image_id ON image_meta(image_id)")
        self.insert(*(row(f"id-{index}", f"{index:03d}.png", vector(2), "general" if index % 2 else "") for index in range(50)))
        finalize_db(self.connection)
        finalize_db(self.connection)
        indexes = {info[1] for info in self.connection.execute("PRAGMA index_list(image_meta)")}
        self.assertLessEqual({"idx_image_meta_path", "idx_image_meta_rating"}, indexes)
        self.assertNotIn("idx_image_meta_image_id", indexes)

        def plan(statement):
            return " ".join(step[3] for step in self.connection.execute("EXPLAIN QUERY PLAN " + statement))

        histogram = (
            "SELECT CASE WHEN rating IN ('general','sensitive','questionable','explicit') THEN rating "
            "ELSE 'unclassified' END, COUNT(*) FROM image_meta GROUP BY 1"
        )
        self.assertIn("USING COVERING INDEX idx_image_meta_rating", plan(histogram))
        self.assertIn("idx_image_meta_path", plan("SELECT image_id FROM image_meta ORDER BY image_path, image_id LIMIT 60"))

        self.connection.close()
        (self.directory / "metafiles.index").write_bytes(b"unread")
        catalog = CatalogDatabase(ModelRegistry(self.root))
        model = ModelId("Test", "model")
        self.assertEqual((catalog.count(model), catalog.count(model, ("general",)), catalog.count(model, ("unclassified",))), (50, 25, 25))
        self.assertEqual([item.name for item in catalog.page(model, 0, 3)], ["000.png", "001.png", "002.png"])


class ImageScanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "images"
        for name in (
            "b.PNG", "a.jpg", "notes.txt", "sub/c.webp", "sub/deep/山の写真.jpeg", "sub/deep/skip.json",
            "空 白/d e.gif", "folder.png/inside.bmp",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        (self.root / "empty").mkdir()

    def scan_result(self, root=None, extensions=(".png", "jpg", ".JPEG", ".webp", ".gif", ".bmp"), **options):
        with mock.patch.object(image_scan.tqdm, "tqdm"), redirect_stdout(io.StringIO()):
            return scan_image_dir(root or self.root, extensions, **options)

    def scan(self, root=None, **options):
        return self.scan_result(root, **options).paths

    def scan_while_listing_sub(self, behave, **options):
        """Scan while listing the subdirectory ``sub`` behaves as the test asks."""
        scandir = os.scandir

        def patched(path):
            if Path(path) == self.root / "sub":
                behave()
            return scandir(path)

        with mock.patch.object(image_scan.os, "scandir", side_effect=patched):
            return self.scan_result(**options)

    def test_images_are_listed_recursively_as_sorted_relative_paths(self):
        expected = sorted(
            os.path.relpath(path, self.root) for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
        )
        self.assertEqual(len(expected), 6)
        self.assertEqual(self.scan(), expected)
        self.assertIn(os.path.join("sub", "deep", "山の写真.jpeg"), expected)

    def test_relative_paths_and_ids_do_not_depend_on_how_the_root_is_written(self):
        expected = self.scan()
        original = os.getcwd()
        os.chdir(self.root.parent)
        try:
            for root in ("images", "./images", os.path.join("images", "sub", ".."), str(self.root) + os.sep):
                with self.subTest(root=root):
                    self.assertEqual(self.scan(root), expected)
        finally:
            os.chdir(original)
        self.assertEqual(image_id_for_path("a.jpg"), hashlib.sha256(b"a.jpg").hexdigest())

    def test_only_requested_extensions_are_listed_and_missing_roots_are_empty(self):
        self.assertEqual(self.scan(extensions=("gif",)), [os.path.join("空 白", "d e.gif")])
        self.assertEqual(self.scan_result().unreadable, [])
        # Unreadable places are named like image paths, relative to the root; the root itself is "".
        self.assertEqual(self.scan_result(self.root / "missing"), ImageScan([], [""]))

    def test_directories_that_cannot_be_listed_are_reported_and_the_rest_is_scanned(self):
        def deny():
            raise PermissionError(13, "Access is denied")

        def network_error():
            # Windows reports a dropped share as a plain OSError.
            raise OSError(22, "The specified network name is no longer available")

        for failure in (deny, network_error):
            with self.subTest(failure=failure.__name__):
                result = self.scan_while_listing_sub(failure)
                self.assertEqual(result.unreadable, ["sub"])
                self.assertEqual(result.paths, [path for path in self.scan() if not path.startswith("sub")])
                self.assertEqual(len(result.paths), 4)

    def test_an_entry_whose_type_cannot_be_determined_is_reported(self):
        scandir = os.scandir
        target = self.root / "sub"

        def cannot_tell(**_):
            raise PermissionError(13, "Permission denied")

        def patched(path):
            if Path(path) != target:
                return scandir(path)
            with scandir(path) as entries:
                odd = types.SimpleNamespace(name="odd", path=os.path.join(target, "odd"), is_dir=cannot_tell)
                return contextlib.nullcontext([*entries, odd])

        with mock.patch.object(image_scan.os, "scandir", side_effect=patched):
            result = self.scan_result()
        # It may be a directory whose images were therefore not listed.
        self.assertEqual(result.unreadable, [os.path.join("sub", "odd")])
        self.assertEqual(len(result.paths), 6)

    def test_a_directory_that_stalls_is_given_up_and_reported(self):
        # The limit applies to every directory; keep it well above what listing a small one takes on a busy machine.
        result = self.scan_while_listing_sub(lambda: time.sleep(1.5), scan_timeout_sec=0.5)
        self.assertEqual(result.unreadable, ["sub"])
        self.assertEqual(len(result.paths), 4)


if __name__ == "__main__":
    unittest.main()
