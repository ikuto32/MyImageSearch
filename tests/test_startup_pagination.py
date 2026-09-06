from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.application.accessor import Accessor
from app.domain.domain_object import ModelId
from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.model_metadata import file_sha256, open_readonly_database


class StartupPaginationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index_dir = self.root / "Test-model"
        self.index_dir.mkdir()
        self.db_path = self.index_dir / "sqlite_image_meta.db"
        (self.index_dir / "metafiles.index").write_bytes(b"FAISS is deliberately unavailable")
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("CREATE TABLE image_meta (image_id TEXT PRIMARY KEY, image_path TEXT, image_tags TEXT)")
            con.execute("CREATE INDEX idx_image_meta_path ON image_meta(image_path)")
            con.executemany("INSERT INTO image_meta VALUES (?, ?, ?)", (
                (str(index), f"image_{index:08d}.png", "tag") for index in range(10_000)
            ))
            con.commit()
        self.model_id = ModelId("Test", "model")
        self.accessor = LocalAccessor(self.root)

    def test_default_accessor_preserves_legacy_fallback(self):
        self.assertIsNone(Accessor().load_startup_image_page(self.model_id, 0, 60))
        self.assertIsNone(self.accessor.load_startup_image_page(ModelId("missing", ""), 0, 60))

    def test_page_reads_are_bounded_readonly_and_do_not_load_faiss(self):
        queries = []

        def connect(path):
            con = open_readonly_database(path)
            con.set_trace_callback(queries.append)
            return con

        before = file_sha256(self.db_path)
        with patch.object(self.accessor, "_load_search_index", side_effect=AssertionError("FAISS must not load")):
            with patch("app.infrastructure.local_accessor.open_readonly_database", side_effect=connect):
                items, paths = self.accessor.load_startup_image_page(self.model_id, 2, 60)
        self.assertEqual(len(items), 60)
        self.assertEqual(len(paths), 60)
        self.assertEqual([item.id.id for item in items], [str(index) for index in range(120, 180)])
        selects = [query for query in queries if query.lstrip().upper().startswith("SELECT")]
        self.assertEqual(len(selects), 1)
        self.assertIn("LIMIT 60 OFFSET 120", selects[0])
        self.assertNotIn("meta,", selects[0])
        self.assertEqual(file_sha256(self.db_path), before)

    def test_duplicate_paths_have_stable_primary_key_tiebreak_across_pages(self):
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("DELETE FROM image_meta")
            con.executemany("INSERT INTO image_meta VALUES (?, ?, NULL)", [
                ("c", "same.png"), ("b", "same.png"), ("a", "same.png"), ("z", "z.png")
            ])
            con.commit()
        first, _ = self.accessor.load_startup_image_page(self.model_id, 0, 2)
        second, _ = self.accessor.load_startup_image_page(self.model_id, 1, 2)
        self.assertEqual([item.id.id for item in first + second], ["a", "b", "c", "z"])
        self.assertTrue(all(item.tags.tags == "" for item in first + second))

    def test_last_page_and_offsets_larger_than_sqlite_range_are_empty(self):
        self.assertEqual(self.accessor.load_startup_image_page(self.model_id, 1000, 60), ([], {}))
        self.assertEqual(self.accessor.load_startup_image_page(self.model_id, 10**100, 60), ([], {}))

    def test_invalid_page_values_fail_before_querying(self):
        for page, size in [(-1, 60), (0, 0), (True, 60), (0, 1.5)]:
            with self.subTest(page=page, size=size):
                with self.assertRaises(ValueError):
                    self.accessor.load_startup_image_page(self.model_id, page, size)

    def test_pagination_uses_the_existing_path_index(self):
        with closing(open_readonly_database(self.db_path)) as con:
            plan = con.execute("""
                EXPLAIN QUERY PLAN SELECT image_id, image_path, image_tags
                FROM image_meta ORDER BY image_path, image_id LIMIT 60 OFFSET 0
            """).fetchall()
        self.assertTrue(any("idx_image_meta_path" in row[3] for row in plan), plan)
        self.assertFalse(any(row[3] == "SCAN image_meta" for row in plan), plan)


if __name__ == "__main__":
    unittest.main()
