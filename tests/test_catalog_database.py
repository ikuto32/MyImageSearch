from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from app.domain.domain_object import ImageId, ModelId
from app.domain.errors import SearchInputError
from app.domain.rating import normalize_rating_filter
from app.infrastructure.catalog_database import CatalogDatabase
from app.infrastructure.index_files import file_sha256, open_readonly_database, readonly_database_uri
from app.infrastructure.model_registry import ModelRegistry
from helpers import create_catalog, traced_connections

OPEN_DATABASE = "app.infrastructure.catalog_database.open_readonly_database"
MIXED_ROWS = [
    ("z", "z.png", "z", "general"), ("b", "a.png", "b", None),
    ("a", "a.png", "a", "general"), ("c", "b.png", "c", "safe"),
    ("d", "c.png", "d", ""), ("e", "d.png", "e", "explicit"),
]


class ReadOnlyDatabaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # Characters with a meaning in a URI must not change which file is opened.
        self.files = create_catalog(Path(temporary.name) / "メタ dir #1 100%25", "Test-model", [("a", "a.png", "", "")])

    def test_source_databases_cannot_be_written_or_created(self):
        missing = self.files.directory / "missing.db"
        with closing(open_readonly_database(self.files.database)) as connection:
            self.assertEqual(connection.execute("SELECT image_id FROM image_meta").fetchall(), [("a",)])
            with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                connection.execute("DELETE FROM image_meta")
            connection.execute("ATTACH DATABASE ? AS manifest", (readonly_database_uri(self.files.database),))
            with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                connection.execute("DELETE FROM manifest.image_meta")
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("ATTACH DATABASE ? AS gone", (readonly_database_uri(missing),))
        with self.assertRaises(sqlite3.OperationalError):
            open_readonly_database(missing).close()
        self.assertFalse(missing.exists())

    def test_names_that_are_not_valid_unicode_are_a_database_error(self):
        # A lone surrogate stands for a byte the file system could not decode.
        with self.assertRaises(sqlite3.OperationalError):
            open_readonly_database(self.files.directory / "lone\udc80surrogate" / "sqlite_image_meta.db").close()

    @unittest.skipUnless(os.name == "nt", "UNC paths exist only on Windows")
    def test_network_paths_are_opened_with_an_empty_uri_authority(self):
        # SQLite rejects file://server/share/...; the server name has to stay in the path.
        uri = readonly_database_uri(Path(r"\\localhost\C$\meta dir\sqlite_image_meta.db"))
        self.assertEqual(uri, "file:////localhost/C%24/meta%20dir/sqlite_image_meta.db?mode=ro")


class CatalogDatabaseTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def catalog(self) -> CatalogDatabase:
        """Call after creating every index directory: the registry scans once."""
        return CatalogDatabase(ModelRegistry(self.root))

    @staticmethod
    def ids(items):
        return [item.id for item in items]


class RatingFilterTests(CatalogDatabaseTestCase):
    def setUp(self):
        super().setUp()
        self.model = ModelId("Main", "weights")
        create_catalog(self.root, "Main-weights", MIXED_ROWS)

    def test_filtered_pages_and_counts_share_the_same_normalization_and_order(self):
        catalog = self.catalog()
        for ratings, expected in (
            (None, ["a", "b", "c", "d", "e", "z"]), ([], []), (["general"], ["a", "z"]),
            (["unclassified"], ["b", "c", "d"]), (["explicit", "general"], ["a", "e", "z"]),
        ):
            with self.subTest(ratings=ratings):
                selected = normalize_rating_filter(ratings)
                count = catalog.count(self.model, selected)
                actual = []
                for page in range((count + 1) // 2):
                    actual.extend(self.ids(catalog.page(self.model, page, 2, selected)))
                self.assertEqual(count, len(expected))
                self.assertEqual(actual, expected)
                self.assertEqual(self.ids(catalog.page(self.model, 0, 1024, selected)), expected)

    def test_first_rows_are_sorted_and_filtered_before_limiting(self):
        catalog = self.catalog()
        self.assertEqual(self.ids(catalog.page(self.model, 0, 3)), ["a", "b", "c"])
        self.assertEqual(self.ids(catalog.page(self.model, 0, 2, ("unclassified",))), ["b", "c"])
        self.assertEqual(self.ids(catalog.page(self.model, 0, 1, ("general",))), ["a"])
        self.assertEqual(catalog.page(self.model, 0, 1024, ()), [])

    def test_all_rating_combinations_share_one_sql_histogram_and_models_remain_independent(self):
        create_catalog(self.root, "Other-weights", [("other", "other.png", "", "general")])
        catalog = self.catalog()
        queries = []
        with patch(OPEN_DATABASE, side_effect=traced_connections(queries)):
            self.assertEqual(catalog.count(self.model, ("general",)), 2)
            self.assertEqual(catalog.count(self.model, ("explicit", "general")), 3)
            self.assertEqual(catalog.count(self.model, ("unclassified",)), 3)
            self.assertEqual(catalog.count(self.model, ()), 0)
            self.assertEqual(catalog.count(self.model), 6)
        self.assertEqual(len([query for query in queries if "COUNT(*)" in query.upper()]), 1)
        self.assertEqual(catalog.count(ModelId("Other", "weights"), ("general",)), 1)

    def test_concurrent_rating_changes_share_a_single_initial_aggregate(self):
        catalog = self.catalog()
        aggregates = []

        def connect(path):
            connection = open_readonly_database(path)

            def trace(query):
                if "GROUP BY" in query:
                    aggregates.append(query)
                    time.sleep(0.03)
            connection.set_trace_callback(trace)
            return connection

        ratings = [("general",), ("explicit",), ("unclassified",), ("sensitive",), ("explicit", "general")]
        with patch(OPEN_DATABASE, side_effect=connect):
            with ThreadPoolExecutor(max_workers=5) as pool:
                counts = list(pool.map(lambda value: catalog.count(self.model, value), ratings))
        self.assertEqual(counts, [2, 1, 3, 0, 3])
        self.assertEqual(len(aggregates), 1)

    def test_database_without_rating_column_is_only_unclassified(self):
        create_catalog(self.root, "Legacy-weights", [("old", "old.png", "")], ("image_id", "image_path", "image_tags"))
        catalog = self.catalog()
        model = ModelId("Legacy", "weights")
        self.assertEqual(catalog.count(model, ("unclassified",)), 1)
        self.assertEqual(catalog.count(model, ("general",)), 0)
        self.assertEqual(catalog.page(model, 0, 60, ("general",)), [])
        self.assertEqual(catalog.page(model, 0, 60, ("unclassified",))[0].rating, "")

    def test_malformed_ratings_are_rejected_when_normalized(self):
        for ratings in ("general", [None], ["unknown"], [{}]):
            with self.subTest(ratings=ratings), self.assertRaises(SearchInputError):
                normalize_rating_filter(ratings)
        self.assertIsNone(normalize_rating_filter(["unclassified", "general", "explicit", "sensitive", "questionable"]))
        self.assertEqual(normalize_rating_filter(["general", "explicit", "general"]), ("explicit", "general"))


class PagingTests(CatalogDatabaseTestCase):
    def test_page_reads_are_bounded_and_readonly(self):
        files = create_catalog(self.root, "Test-model", [
            (str(index), f"image_{index:08d}.png", "tag", None) for index in range(10_000)
        ])
        catalog = self.catalog()
        queries = []
        before = file_sha256(files.database)
        with patch(OPEN_DATABASE, side_effect=traced_connections(queries)):
            items = catalog.page(ModelId("Test", "model"), 2, 60)
        self.assertEqual(self.ids(items), [str(index) for index in range(120, 180)])
        self.assertEqual(items[0].name, "image_00000120.png")
        selects = [query for query in queries if query.lstrip().upper().startswith("SELECT")]
        self.assertEqual(len(selects), 1)
        self.assertIn("LIMIT 60 OFFSET 120", selects[0])
        # Only the listed columns are read; embeddings and tags stay on disk.
        self.assertIn("SELECT image_id, image_path, rating", " ".join(selects[0].split()))
        self.assertEqual(file_sha256(files.database), before)

    def test_duplicate_paths_have_stable_primary_key_tiebreak_across_pages(self):
        create_catalog(self.root, "Test-model", [
            ("c", "same.png", None, None), ("b", "same.png", None, None),
            ("a", "same.png", None, None), ("z", "z.png", None, None),
        ])
        catalog = self.catalog()
        model = ModelId("Test", "model")
        items = catalog.page(model, 0, 2) + catalog.page(model, 1, 2)
        self.assertEqual(self.ids(items), ["a", "b", "c", "z"])
        self.assertTrue(all(item.tags == "" for item in items))

    def test_pages_past_the_end_and_offsets_beyond_sqlite_range_are_empty(self):
        create_catalog(self.root, "Test-model", [("a", "a.png", "", "")])
        catalog = self.catalog()
        model = ModelId("Test", "model")
        self.assertEqual(catalog.page(model, 1000, 60), [])
        self.assertEqual(catalog.page(model, 10**100, 60), [])

    def test_invalid_page_values_and_unregistered_models_fail_before_querying(self):
        create_catalog(self.root, "Test-model", [("a", "a.png", "", "")])
        catalog = self.catalog()
        with patch(OPEN_DATABASE, side_effect=AssertionError("must validate first")):
            for page, size in [(-1, 60), (0, 0), (True, 60), (0, 1.5)]:
                with self.subTest(page=page, size=size), self.assertRaises(ValueError):
                    catalog.page(ModelId("Test", "model"), page, size)
            with self.assertRaises(SearchInputError):
                catalog.page(ModelId("missing", ""), 0, 60)
            with self.assertRaises(SearchInputError):
                catalog.count(ModelId("missing", ""))

    def test_pagination_uses_the_path_index(self):
        files = create_catalog(self.root, "Test-model", [("a", "a.png", "", "")])
        with closing(open_readonly_database(files.database)) as connection:
            plan = connection.execute("""
                EXPLAIN QUERY PLAN SELECT image_id, image_path, rating
                FROM image_meta ORDER BY image_path ASC, image_id ASC LIMIT 60 OFFSET 0
            """).fetchall()
        self.assertTrue(any("idx_image_meta_path" in row[3] for row in plan), plan)
        self.assertFalse(any(row[3] == "SCAN image_meta" for row in plan), plan)

    def test_arbitrary_filtered_page_can_be_loaded_without_reading_prior_pages(self):
        create_catalog(self.root, "Other-weights", [
            (str(i), f"{i:05d}.png", "tag", "general" if i % 2 else "explicit") for i in range(1000)
        ])
        catalog = self.catalog()
        model = ModelId("Other", "weights")
        self.assertEqual(catalog.count(model, ("general",)), 500)
        items = catalog.page(model, 8, 60, ("general",))
        self.assertEqual(self.ids(items), [str(i) for i in range(961, 1000, 2)])
        self.assertTrue(all(item.rating == "general" for item in items))

    def test_covering_path_seek_preserves_duplicate_path_and_null_boundary_order(self):
        rows = [("null-a", None, "", "general"), ("null-b", None, "", "general")]
        rows += [(f"a-{i}", f"a{i:04d}.png", "", "general") for i in range(2000)]
        rows += [(f"dup-{1000 - i:04d}", "middle.png", "", "general") for i in range(1000)]
        rows += [(f"z-{i}", f"z{i:04d}.png", "", "general") for i in range(2000)]
        files = create_catalog(self.root, "Large-weights", rows)
        catalog = self.catalog()
        model = ModelId("Large", "weights")
        self.assertEqual(catalog.count(model), len(rows))
        with closing(open_readonly_database(files.database)) as connection:
            expected = [row[0] for row in connection.execute(
                "SELECT image_id FROM image_meta ORDER BY image_path, image_id LIMIT 60 OFFSET 2220"
            )]
        queries = []
        with patch(OPEN_DATABASE, side_effect=traced_connections(queries)):
            self.assertEqual(self.ids(catalog.page(model, 37, 60)), expected)
        self.assertTrue(any("image_path >=" in query for query in queries), queries)
        self.assertEqual(self.ids(catalog.page(model, 0, 2)), ["null-a", "null-b"])
        self.assertEqual(catalog.page(model, 1000, 60), [])

    def test_tail_pages_are_read_from_the_end_once_the_total_is_known(self):
        create_catalog(self.root, "Deep-weights", [(str(i), f"{i:05d}.png", "", "general") for i in range(12000)])
        catalog = self.catalog()
        model = ModelId("Deep", "weights")
        self.assertEqual(catalog.count(model), 12000)
        queries = []
        with patch(OPEN_DATABASE, side_effect=traced_connections(queries)):
            items = catalog.page(model, 199, 60)
        self.assertEqual(self.ids(items), [str(i) for i in range(11940, 12000)])
        page_queries = [query for query in queries if "LIMIT" in query]
        self.assertEqual(len(page_queries), 1)
        self.assertIn("DESC", page_queries[0])
        self.assertIn("OFFSET 0", page_queries[0])

    def test_adjacent_deep_filtered_pages_seek_from_a_recent_boundary(self):
        create_catalog(self.root, "Deep-weights", [(str(i), f"{i:05d}.png", "", "general") for i in range(12000)])
        catalog = self.catalog()
        model = ModelId("Deep", "weights")
        self.assertEqual(catalog.count(model, ("general",)), 12000)
        catalog.page(model, 100, 60, ("general",))
        for page, first, comparison in ((101, 6060, ">="), (99, 5940, "<=")):
            with self.subTest(page=page):
                queries = []
                with patch(OPEN_DATABASE, side_effect=traced_connections(queries)):
                    items = catalog.page(model, page, 60, ("general",))
                self.assertEqual(self.ids(items), [str(i) for i in range(first, first + 60)])
                page_queries = [query for query in queries if "LIMIT" in query]
                self.assertEqual(len(page_queries), 1)
                self.assertIn(f"(image_path, image_id) {comparison}", page_queries[0])
                self.assertIn("OFFSET 1", page_queries[0])


class LookupTests(CatalogDatabaseTestCase):
    COLUMNS = ("image_id", "image_path", "image_tags", "rating", "style_cluster", "aesthetic_quality")

    def test_find_reads_one_row_with_its_details(self):
        create_catalog(self.root, "Test-model", [
            ("a", "a.png", None, None, None, None), ("b", "dir/b.png", "blue", "general", "style_2", 6.5),
        ], self.COLUMNS)
        catalog = self.catalog()
        model = ModelId("Test", "model")
        found = catalog.find(model, ImageId("b"))
        self.assertEqual(
            (found.id, found.name, found.tags, found.rating, found.style_cluster, found.aesthetic_quality),
            ("b", "dir/b.png", "blue", "general", "style_2", 6.5),
        )
        unset = catalog.find(model, ImageId("a"))
        self.assertEqual((unset.tags, unset.rating, unset.style_cluster, unset.aesthetic_quality), ("", "", "", None))
        self.assertIsNone(catalog.find(model, ImageId("missing")))
        self.assertEqual(catalog.find_path(model, ImageId("b")), "dir/b.png")
        self.assertIsNone(catalog.find_path(model, ImageId("missing")))

    def test_older_optional_metadata_columns_are_returned_as_unset(self):
        create_catalog(self.root, "Test-model", [("a", "a.png", 4.0)], ("image_id", "image_path", "aesthetic_quality"))
        found = self.catalog().find(ModelId("Test", "model"), ImageId("a"))
        self.assertEqual((found.tags, found.rating, found.style_cluster), ("", "", ""))
        self.assertEqual(found.aesthetic_quality, 4.0)

    def test_database_without_required_columns_is_rejected(self):
        create_catalog(self.root, "Test-model", [("a",)], ("image_id",), path_index=False)
        with self.assertRaisesRegex(ValueError, "image_id and image_path"):
            self.catalog().page(ModelId("Test", "model"), 0, 60)


if __name__ == "__main__":
    unittest.main()
