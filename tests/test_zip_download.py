from contextlib import closing
import errno
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from PIL import Image as PILImage

from app.application.usecase import Usecase
from app.domain.domain_object import Image, ImageId, ImageName, ModelId
from app.domain.errors import ResourceLimitError, SearchInputError
from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.local_repository import LocalRepository
from app.presentation import controller


class CatalogDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.meta = self.root / "meta"
        self.meta.mkdir()
        self.accessor = LocalAccessor(self.meta)
        self.repo = LocalRepository(self.images, self.meta)
        self.model = ModelId("Main", "weights")

    def catalog(self, name, rows, *, with_rating=True):
        directory = self.meta / name
        directory.mkdir()
        (directory / "metafiles.index").write_bytes(b"unread fake index")
        with closing(sqlite3.connect(directory / "sqlite_image_meta.db")) as connection:
            suffix = ", rating TEXT" if with_rating else ""
            connection.execute(f"CREATE TABLE image_meta (image_id TEXT PRIMARY KEY, image_path TEXT, image_tags TEXT{suffix})")
            connection.execute("CREATE INDEX image_path_order ON image_meta(image_path, image_id)")
            marks = "?,?,?,?" if with_rating else "?,?,?"
            connection.executemany(f"INSERT INTO image_meta VALUES ({marks})", rows)
            connection.commit()

    def test_filters_unknown_categories_and_sorts_before_limiting_without_index(self):
        self.catalog("Main-weights", [
            ("z", "z.png", "", "general"),
            ("b", "a.png", "", None),
            ("a", "a.png", "", "general"),
            ("c", "b.png", "", "old-safe"),
            ("d", "c.png", "", ""),
            ("e", "d.png", "", "explicit"),
        ])
        with patch.object(self.accessor, "load_index_with_metadata", side_effect=AssertionError("must not load index")), patch.object(self.accessor, "load_image_metadata", side_effect=AssertionError("must not load all metadata")):
            self.assertEqual([i.id.id for i in self.accessor.load_download_image_items(self.model, 3)], ["a", "b", "c"])
            self.assertEqual([i.id.id for i in self.accessor.load_download_image_items(self.model, 2, ["unclassified"])], ["b", "c"])
            self.assertEqual([i.id.id for i in self.accessor.load_download_image_items(self.model, 1, ["general"])], ["a"])
            self.assertEqual(self.accessor.load_download_image_items(self.model, 1024, []), [])
        self.assertEqual(self.accessor._search_indexes, {})

    def test_legacy_missing_rating_column_matches_only_unclassified(self):
        self.catalog("Main-weights", [("a", "a.png", "")], with_rating=False)
        self.assertEqual(len(self.accessor.load_download_image_items(self.model, 5, ["unclassified"])), 1)
        self.assertEqual(self.accessor.load_download_image_items(self.model, 5, ["general"]), [])

    def test_selected_model_is_used_and_its_paths_are_registered(self):
        self.catalog("Main-weights", [("main", "a.png", "", "general")])
        self.catalog("Other-weights", [("other", "b.png", "", "general")])
        usecase = Usecase(self.repo, self.accessor, self.model)
        self.assertEqual(usecase.get_download_ids(ModelId("Other", "weights"), 1024, ["general"]), ["other"])
        self.assertEqual(self.repo.get_image_name(ImageId("other")).name, "b.png")
        with self.assertRaises(SearchInputError):
            self.accessor.load_download_image_items(ModelId("Missing", "weights"), 1)

    def test_limit_boundary_and_invalid_categories(self):
        self.catalog("Main-weights", [(str(i), f"{i:04d}.png", "", "general") for i in range(1030)])
        result = self.accessor.load_download_image_items(self.model, 1024)
        self.assertEqual(len(result), 1024)
        self.assertEqual(result[-1].display_name.name, "1023.png")
        for limit in (0, -1, 1025, True):
            with self.subTest(limit=limit), self.assertRaises(SearchInputError):
                self.accessor.load_download_image_items(self.model, limit)
        for ratings in ("general", ["unknown"], [None], [{}]):
            with self.subTest(ratings=ratings), self.assertRaises(SearchInputError):
                self.accessor.load_download_image_items(self.model, 1, ratings)

    def test_missing_files_are_counted_in_http_and_empty_archives_fail(self):
        self.catalog("Main-weights", [("a", "a.png", "", "general"), ("b", "missing.png", "", "general")])
        PILImage.new("RGB", (10, 10), "red").save(self.images / "a.png")
        usecase = Usecase(self.repo, self.accessor, self.model)
        with patch.object(controller, "usecase", usecase):
            response = controller.app.test_client().post("/download_images_zip", json={"params": {"first": 2, "model_name": "Main", "pretrained": "weights"}})
            self.addCleanup(response.close)
            self.assertEqual(response.status_code, 200, response.data)
            self.assertEqual(response.headers["X-Archive-Requested"], "2")
            self.assertEqual(response.headers["X-Archive-Images"], "1")
            self.assertEqual(response.headers["X-Archive-Skipped"], "1")
            with zipfile.ZipFile(BytesIO(response.data)) as archive:
                self.assertEqual(archive.namelist(), ["a.png"])
            response = controller.app.test_client().post("/download_images_zip", json={"params": {"ids": ["b"]}})
            self.assertEqual(response.status_code, 400)
            self.assertIn("保存できる画像", response.get_json()["error"])


class DownloadBoundaryTests(unittest.TestCase):
    def test_post_accepts_1024_and_rejects_1025_before_archive_creation(self):
        fake = Mock()
        fake.get_images_zip.return_value = BytesIO(b"archive")
        with patch.object(controller, "usecase", fake):
            client = controller.app.test_client()
            response = client.post("/download_images_zip", json={"params": {"ids": [str(i) for i in range(1024)]}})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, b"archive")
            response.close()
            fake.reset_mock()
            for params in (
                {"ids": [str(i) for i in range(1025)]},
                {"first": 1025, "model_name": "Main"},
                {"first": 1, "ids": ["a"], "model_name": "Main"},
                {"first": 1, "model_name": "Main", "ratings": ["unknown"]},
            ):
                response = client.post("/download_images_zip", json={"params": params})
                self.assertEqual(response.status_code, 400, response.data)
            fake.get_images_zip.assert_not_called()
            fake.get_download_ids.assert_not_called()

    def test_archive_writes_all_1024_entries_with_bounded_spool(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LocalRepository(Path(directory))
            data = Image(b"image contents", "image/png")
            entries = ((data, ImageName(f"{i:04d}.png")) for i in range(1024))
            with repo.create_zip_from_images(entries) as stream:
                with zipfile.ZipFile(stream) as archive:
                    self.assertEqual(len(archive.infolist()), 1024)
                    self.assertEqual(archive.namelist()[-1], "1023.png")

    def test_disk_capacity_failure_is_detected_before_member_write_and_closes_spool(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LocalRepository(Path(directory))
            spool = tempfile.SpooledTemporaryFile(max_size=1)
            with patch.object(repo, "ZIP_MEMORY_BYTES", 1), patch("app.infrastructure.local_repository.tempfile.SpooledTemporaryFile", return_value=spool), patch("app.infrastructure.local_repository.shutil.disk_usage", return_value=SimpleNamespace(free=0)), patch("app.infrastructure.local_repository.zipfile.ZipFile.writestr") as write:
                with self.assertRaises(ResourceLimitError):
                    repo.create_zip_from_images([(Image(b"image", "image/png"), ImageName("a.png"))])
            write.assert_not_called()
            self.assertTrue(spool.closed)

    def test_disk_full_race_becomes_resource_error_and_closes_spool(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LocalRepository(Path(directory))
            spool = tempfile.SpooledTemporaryFile()
            with patch("app.infrastructure.local_repository.tempfile.SpooledTemporaryFile", return_value=spool), patch("app.infrastructure.local_repository.zipfile.ZipFile.writestr", side_effect=OSError(errno.ENOSPC, "disk full")):
                with self.assertRaises(ResourceLimitError):
                    repo.create_zip_from_images([(Image(b"image", "image/png"), ImageName("a.png"))])
            self.assertTrue(spool.closed)


if __name__ == "__main__":
    unittest.main()
