import unittest
from contextlib import closing
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import Mock, patch
import zipfile

from PIL import Image as PILImage

from app.application.usecase import Usecase
from app.domain.domain_object import ImageId, ImageItem, ImageName, ModelId
from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.local_repository import LocalRepository
from app.presentation import controller


class PagedStartupTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelId('model', 'weights')
        self.items = [ImageItem(ImageId(str(i)), ImageName(f'{i:03d}.jpg')) for i in range(123)]
        self.accessor = Mock()
        self.repository = Mock()
        self.repository.load_all_model_item.return_value = []

        def page(model, number, size):
            self.assertEqual(model, self.model)
            items = self.items[number * size:(number + 1) * size]
            return items, {item.id: Path(item.display_name.name) for item in items}

        self.accessor.load_startup_image_page.side_effect = page

    def test_startup_and_pages_read_only_requested_rows_and_keep_previous_images(self):
        usecase = Usecase(self.repository, self.accessor, self.model)
        self.accessor.load_startup_image_page.assert_called_once_with(self.model, 0, 60)
        self.accessor.load_startup_image_items.assert_not_called()
        self.repository.load_all_image_item.assert_not_called()
        self.accessor.load_index_with_metadata.assert_not_called()
        self.assertEqual(len(usecase._id_to_image_items), 60)
        self.assertEqual(usecase._image_items, [])
        self.assertEqual(usecase.get_image_items_by_page(2, 60), self.items[120:])
        self.repository.register_image_items.assert_called_with(self.items[120:])
        self.assertEqual(usecase.get_image_item(self.items[0].id), self.items[0])
        self.assertEqual(usecase.get_image_item(self.items[-1].id), self.items[-1])
        self.assertEqual(len(usecase._id_to_image_items), 63)
        self.assertEqual(usecase.get_image_items_by_page(3, 60), [])
        self.assertEqual(usecase.get_image_items_by_page(-1, 5), self.items[:5])

    def test_empty_database_does_not_trigger_filesystem_scan(self):
        self.items.clear()
        usecase = Usecase(self.repository, self.accessor, self.model)
        self.assertEqual(usecase.get_image_items_by_page(0, 60), [])
        self.repository.load_all_image_item.assert_not_called()
        self.accessor.load_startup_image_items.assert_not_called()

    def test_legacy_accessor_falls_back_and_retains_paging(self):
        self.accessor.load_startup_image_page.side_effect = None
        self.accessor.load_startup_image_page.return_value = None
        self.accessor.load_startup_image_items.return_value = (self.items, {})
        usecase = Usecase(self.repository, self.accessor, self.model)
        self.assertEqual(usecase.get_image_items_by_page(2, 60), self.items[120:])
        self.assertEqual(usecase.get_all_image_item(), self.items)

    def test_all_items_are_loaded_only_for_explicit_legacy_call(self):
        self.accessor.load_startup_image_items.return_value = (self.items, {})
        usecase = Usecase(self.repository, self.accessor, self.model)
        self.assertEqual(usecase.get_all_image_item(), self.items)
        self.accessor.load_startup_image_items.assert_called_once_with(self.model)

    def test_saved_image_ids_resolve_without_browsing_their_page(self):
        usecase = Usecase(self.repository, self.accessor, self.model)
        target = self.items[-1]
        self.accessor.load_image_metadata.return_value = [target]
        self.assertEqual(usecase.get_image_item(target.id), target)
        self.accessor.load_image_metadata.assert_called_once_with(self.model, [target.id])
        usecase.get_image(target.id)
        usecase.get_small_image(target.id)
        self.repository.load_image.assert_called_once_with(target.id)
        self.repository.load_small_image.assert_called_once_with(target.id)
        self.accessor.load_startup_image_page.assert_called_once()
        self.accessor.load_index_with_metadata.assert_not_called()

    def test_unknown_saved_image_ids_fail_without_scanning_files(self):
        usecase = Usecase(self.repository, self.accessor, self.model)
        self.accessor.load_image_metadata.return_value = []
        with self.assertRaises(ValueError):
            usecase.get_small_image(ImageId('missing'))
        self.repository.load_all_image_item.assert_not_called()


class PagedStartupHTTPTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.meta = self.root / "meta"
        self.meta.mkdir()
        self.model = ModelId("Main", "model")
        self.target_id = "image-65"
        self.target_name = "065.png"
        self.other_id = "other-only"
        self.other_name = "other-only.png"
        self._create_catalog("Main-model", [
            (f"image-{index}", f"{index:03d}.png") for index in range(70)
        ])
        self._create_catalog("Other-model", [(self.other_id, self.other_name)])
        PILImage.new("RGB", (20, 10), "red").save(self.images / self.target_name)
        PILImage.new("RGB", (20, 10), "blue").save(self.images / self.other_name)
        self.usecase_patch = patch.object(controller, "usecase", None)
        self.usecase_patch.start()
        self.addCleanup(self.usecase_patch.stop)

    def _create_catalog(self, name, rows):
        directory = self.meta / name
        directory.mkdir()
        (directory / "metafiles.index").write_bytes(b"Must never be read by these routes")
        with closing(sqlite3.connect(directory / "sqlite_image_meta.db")) as con:
            con.execute("CREATE TABLE image_meta (image_id TEXT PRIMARY KEY, image_path TEXT, image_tags TEXT)")
            con.execute("CREATE INDEX idx_image_meta_path ON image_meta(image_path)")
            con.executemany("INSERT INTO image_meta VALUES (?, ?, 'test tag')", rows)
            con.commit()

    def _fresh_client(self):
        accessor = LocalAccessor(self.meta)
        repository = LocalRepository(self.images, self.meta)
        usecase = Usecase(repository, accessor, self.model)
        self.assertEqual(len(usecase._id_to_image_items), 60)
        self.assertNotIn(ImageId(self.target_id), usecase._id_to_image_items)
        self.assertNotIn(ImageId(self.other_id), usecase._id_to_image_items)
        controller.usecase = usecase
        return controller.app.test_client(), usecase, accessor

    def test_cold_saved_item_and_image_urls_work_before_their_page_is_browsed(self):
        for route in (
            f"/image_item/{self.target_id}",
            f"/image/{self.target_id}/small",
            f"/image/{self.target_id}/original",
        ):
            with self.subTest(route=route):
                client, usecase, accessor = self._fresh_client()
                response = client.get(route)
                self.assertEqual(response.status_code, 200, response.data)
                self.assertEqual(len(usecase._id_to_image_items), 61)
                self.assertFalse(accessor._search_indexes)
                self.assertFalse(accessor._embedding_backends)
                if route.startswith("/image_item/"):
                    self.assertEqual(response.get_json()["id"], self.target_id)
                else:
                    self.assertEqual(response.content_type, "image/png")
                    with PILImage.open(BytesIO(response.data)) as image:
                        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_cold_saved_zip_resolves_the_image_without_browsing(self):
        client, usecase, accessor = self._fresh_client()
        response = client.post("/download_images_zip", json={"params": {"ids": [self.target_id]}})
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(BytesIO(response.data)) as archive:
            self.assertEqual(archive.namelist(), [self.target_name])
            self.assertEqual(archive.read(self.target_name), (self.images / self.target_name).read_bytes())
        self.assertEqual(len(usecase._id_to_image_items), 61)
        self.assertFalse(accessor._search_indexes)
        self.assertFalse(accessor._embedding_backends)

    def test_cold_id_present_only_in_another_model_is_resolved(self):
        for route in (f"/image_item/{self.other_id}", f"/image/{self.other_id}/small", f"/image/{self.other_id}/original"):
            with self.subTest(route=route):
                client, usecase, accessor = self._fresh_client()
                response = client.get(route)
                self.assertEqual(response.status_code, 200, response.data)
                self.assertIn(ImageId(self.other_id), usecase._id_to_image_items)
                self.assertFalse(accessor._search_indexes)
                self.assertFalse(accessor._embedding_backends)

    def test_unknown_cold_ids_return_404_without_growing_the_catalog(self):
        for route in ("/image_item/missing", "/image/missing/small", "/image/missing/original"):
            with self.subTest(route=route):
                client, usecase, accessor = self._fresh_client()
                self.assertEqual(client.get(route).status_code, 404)
                self.assertEqual(len(usecase._id_to_image_items), 60)
                self.assertFalse(accessor._search_indexes)
                self.assertFalse(accessor._embedding_backends)


if __name__ == '__main__':
    unittest.main()
