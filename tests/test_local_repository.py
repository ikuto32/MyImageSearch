from io import BytesIO
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from PIL import Image as PILImage

from app.application.usecase import Usecase
from app.domain.domain_object import ImageId, ImageItem, ModelId
from app.domain.errors import ImageUnavailableError
from app.infrastructure import local_repository
from app.infrastructure.local_repository import LocalRepository
from app.infrastructure.model_registry import ModelRegistry
from app.presentation.controller import create_app
from helpers import create_catalog, save_image, traced_connections

OPEN_DATABASE = "app.infrastructure.catalog_database.open_readonly_database"


class LocalRepositoryTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.meta = self.root / "meta"
        self.main = ModelId("Main", "model")
        self.other = ModelId("Other", "model")
        create_catalog(self.meta, "Main-model", [
            (f"image-{index}", f"{index:03d}.png", "test tag", "general") for index in range(70)
        ])
        create_catalog(self.meta, "Other-model", [("other-only", "sub/other-only.png", "", "explicit")])
        save_image(self.images / "065.png", size=(20, 10), color="red")
        save_image(self.images / "sub" / "other-only.png", size=(20, 10), color="blue")
        self.repository = LocalRepository(self.images, ModelRegistry(self.meta))


class ImageResolutionTests(LocalRepositoryTestCase):
    def test_models_come_from_the_registry(self):
        self.assertEqual([item.id for item in self.repository.load_all_model_item()], [self.main, self.other])

    def test_images_resolve_by_id_in_any_registered_model_without_browsing(self):
        for image_id, color in (("image-65", (255, 0, 0)), ("other-only", (0, 0, 255))):
            for load in (self.repository.load_image, self.repository.load_small_image):
                with self.subTest(image_id=image_id, load=load.__name__):
                    with PILImage.open(BytesIO(load(ImageId(image_id)).binary)) as image:
                        self.assertEqual(image.getpixel((0, 0)), color)
        self.assertEqual(self.repository.load_image(ImageId("other-only")).name, "sub/other-only.png")

    def test_unknown_ids_and_missing_files_are_unavailable(self):
        for image_id in ("missing", "image-1"):
            for load in (self.repository.load_image, self.repository.load_small_image):
                with self.subTest(image_id=image_id, load=load.__name__), self.assertRaises(ImageUnavailableError):
                    load(ImageId(image_id))

    def test_listed_and_resolved_paths_are_remembered_instead_of_queried_again(self):
        queries = []
        with patch(OPEN_DATABASE, side_effect=traced_connections(queries)):
            self.repository.load_image_items_by_page(self.main, 1, 60)
            queries.clear()
            self.repository.load_image(ImageId("image-65"))
            self.repository.load_small_image(ImageId("image-65"))
            self.assertEqual(queries, [])
            self.repository.load_image(ImageId("other-only"))
            lookups = len(queries)
            self.assertGreater(lookups, 0)
            self.repository.load_small_image(ImageId("other-only"))
            self.assertEqual(len(queries), lookups)

    def test_the_model_that_resolved_the_last_image_is_searched_first(self):
        queries = []
        with patch(OPEN_DATABASE, side_effect=traced_connections(queries)):
            self.repository.load_image(ImageId("other-only"))
            self.assertEqual(len(queries), 2)
            queries.clear()
            with self.assertRaises(ImageUnavailableError):
                self.repository.load_image(ImageId("missing"))
            self.assertEqual(len(queries), 2)
        self.assertEqual(self.repository._lookup_order, [self.other, self.main])

    def test_search_results_resolve_without_querying_any_database(self):
        self.repository.remember_image_items([
            ImageItem(ImageId("image-65"), "065.png"), ImageItem(ImageId("other-only"), "sub/other-only.png"),
            ImageItem(ImageId("without-path"), ""),
        ])
        queries = []
        with patch(OPEN_DATABASE, side_effect=traced_connections(queries)):
            self.repository.load_small_image(ImageId("image-65"))
            self.repository.load_image(ImageId("other-only"))
            self.assertEqual(queries, [])
            with self.assertRaises(ImageUnavailableError):
                self.repository.load_image(ImageId("without-path"))
            # An item without a path is not remembered; both catalogs are asked for it.
            self.assertEqual(len(queries), 2)

    def test_an_unreadable_model_database_does_not_hide_the_images_of_other_models(self):
        # Sorts before the readable models, so it is asked first.
        broken = create_catalog(self.meta, "Broken-model", [])
        broken.database.write_bytes(b"not a database")
        repository = LocalRepository(self.images, ModelRegistry(self.meta))
        with self.assertLogs("app.infrastructure.local_repository", level="WARNING") as logs:
            self.assertEqual(repository.load_image(ImageId("image-65")).name, "065.png")
            self.assertEqual(repository._lookup_order, [self.main, self.other, ModelId("Broken", "model")])
            self.assertEqual(repository.load_small_image(ImageId("other-only")).name, "sub/other-only.png")
            with self.assertRaises(ImageUnavailableError):
                repository.load_image(ImageId("missing"))
        # Reported once.
        self.assertEqual(len(logs.output), 1)

    def test_remembered_paths_are_bounded(self):
        with patch.object(local_repository, "_MAX_REMEMBERED_PATHS", 10):
            self.repository.load_image_items_by_page(self.main, 0, 60)
            self.assertEqual(list(self.repository._paths), [f"image-{index}" for index in range(50, 60)])
            self.repository.load_image(ImageId("other-only"))
            self.assertEqual(len(self.repository._paths), 10)
            self.assertIn("other-only", self.repository._paths)

    def test_catalog_queries_are_delegated_with_their_filter(self):
        self.assertEqual(self.repository.count_image_items(self.main), 70)
        self.assertEqual(self.repository.count_image_items(self.other, ("general",)), 0)
        self.assertEqual(self.repository.load_image_items_by_page(self.other, 0, 60, ("explicit",))[0].id, "other-only")
        self.assertEqual(self.repository.find_image_item(self.other, ImageId("other-only")).rating, "explicit")
        self.assertIsNone(self.repository.find_image_item(self.main, ImageId("other-only")))


class SavedImageUrlTests(LocalRepositoryTestCase):
    """Bookmarked image URLs and downloads work without FAISS, models or prior browsing."""

    def setUp(self):
        super().setUp()
        # Skipped archive members are reported in the response; keep the test output quiet.
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.accessor = Mock()
        self.client = create_app(Usecase(self.repository, self.accessor)).test_client()

    def tearDown(self):
        self.assertEqual(self.accessor.mock_calls, [])

    def test_cold_image_urls_work_before_their_page_is_browsed(self):
        for image_id, color in (("image-65", (255, 0, 0)), ("other-only", (0, 0, 255))):
            for variant in ("small", "original"):
                with self.subTest(image_id=image_id, variant=variant):
                    response = self.client.get(f"/image/{image_id}/{variant}")
                    self.assertEqual(response.status_code, 200, response.data)
                    self.assertEqual(response.content_type, "image/png")
                    with PILImage.open(BytesIO(response.data)) as image:
                        self.assertEqual(image.getpixel((0, 0)), color)

    def test_unknown_ids_return_404(self):
        for route in ("/image/missing/small", "/image/missing/original", "/image/image-1/small"):
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).status_code, 404)

    def test_selected_images_are_archived_and_missing_files_are_counted(self):
        prepared = self.client.post("/downloads/prepare", json={"params": {"ids": ["image-65", "image-1", "other-only"]}})
        self.assertEqual(prepared.status_code, 200, prepared.data)
        data = prepared.get_json()
        self.assertEqual((data["requested_count"], data["image_count"], data["skipped_count"]), (3, 2, 1))
        response = self.client.get(data["download_url"])
        self.addCleanup(response.close)
        with zipfile.ZipFile(BytesIO(response.data)) as archive:
            self.assertEqual(archive.namelist(), ["065.png", "other-only.png"])
            self.assertEqual(archive.read("065.png"), (self.images / "065.png").read_bytes())

    def test_first_images_of_the_selected_model_are_archived(self):
        params = {"first": 2, "model_name": "Other", "pretrained": "model", "ratings": ["explicit"]}
        data = self.client.post("/downloads/prepare", json={"params": params}).get_json()
        self.assertEqual((data["requested_count"], data["image_count"]), (1, 1))
        response = self.client.get(data["download_url"])
        self.addCleanup(response.close)
        with zipfile.ZipFile(BytesIO(response.data)) as archive:
            self.assertEqual(archive.namelist(), ["other-only.png"])

    def test_archive_without_available_images_is_rejected(self):
        for params in (
            {"ids": ["image-1"]},
            {"first": 5, "model_name": "Main", "pretrained": "model"},
            {"first": 5, "model_name": "Other", "pretrained": "model", "ratings": ["general"]},
        ):
            with self.subTest(params=params):
                response = self.client.post("/downloads/prepare", json={"params": params})
                self.assertEqual(response.status_code, 400)
                self.assertIn("保存できる画像", response.get_json()["error"])
        unknown = self.client.post("/downloads/prepare", json={"params": {"first": 1, "model_name": "Missing"}})
        self.assertEqual(unknown.get_json()["error_code"], "invalid_model_or_image")

    def test_catalog_pages_report_totals_for_the_selected_model(self):
        response = self.client.get("/image_item", query_string={
            "model_name": "Main", "pretrained": "model", "page": 1, "size": 60, "include_total": 1,
        })
        self.assertEqual([item["id"] for item in response.get_json()], [f"image-{index}" for index in range(60, 70)])
        self.assertEqual((response.headers["X-Catalog-Total"], response.headers["X-Matching-Total"]), ("70", "70"))
        detail = self.client.get("/image_meta/other-only", query_string={"model_name": "Other", "pretrained": "model"})
        self.assertEqual(detail.get_json()["rating"], "explicit")


if __name__ == "__main__":
    unittest.main()
