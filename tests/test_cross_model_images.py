import io
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import faiss
import numpy as np
from PIL import Image as PILImage

from app.application.usecase import Usecase
from app.domain.domain_object import ImageId, ImageItem, ImageName, ModelId
from app.infrastructure.local_repository import LocalRepository
from app.presentation import controller


class CrossModelImageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        self.items = [ImageItem(ImageId(name), ImageName(f"{name}.png")) for name in ("startup", "other", "unreturned")]
        for item, color in zip(self.items, ("red", "blue", "green")):
            PILImage.new("RGB", (20, 20), color).save(self.root / item.display_name.name)
        self.repository = LocalRepository(self.root)

    def test_other_model_search_registers_only_returned_images_for_display_detail_and_zip(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.eye(2, dtype=np.float32))
        accessor = Mock()
        accessor.load_startup_image_page.return_value = None
        accessor.load_startup_image_items.return_value = ([self.items[0]], {self.items[0].id: pathlib.Path(self.items[0].display_name.name)})
        accessor.load_index_with_metadata.return_value = (index, self.items[1:])
        accessor.get_mean_meta_vector.return_value = None
        usecase = Usecase(self.repository, accessor, ModelId("startup", ""))
        with patch.object(controller, "usecase", usecase):
            client = controller.app.test_client()
            response = client.post("/search/query", json={"params": {
                "model_name": "other", "search_query": "[1., 0.]", "result_size": 1,
            }})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["list"][0]["item"]["id"], "other")
            self.assertEqual(set(self.repository._id_to_path), {self.items[0].id, self.items[1].id})
            detail = client.get("/image_item/other")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.get_json()["name"], "other.png")
            for variant in ("small", "original"):
                image = client.get(f"/image/other/{variant}")
                self.assertEqual(image.status_code, 200)
                with PILImage.open(io.BytesIO(image.data)) as decoded:
                    self.assertEqual(decoded.getpixel((0, 0)), (0, 0, 255))
            archive = client.post("/download_images_zip", json={"params": {"ids": ["other"]}})
            self.assertEqual(archive.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(archive.data)) as content:
                self.assertEqual(content.namelist(), ["other.png"])
                self.assertEqual(content.read("other.png"), (self.root / "other.png").read_bytes())
            archive.close()
            self.assertEqual(client.get("/image_item/unreturned").status_code, 404)

    def test_registering_changed_path_invalidates_only_its_thumbnail(self):
        self.repository.set_image_paths({item.id: pathlib.Path(item.display_name.name) for item in self.items})
        old = self.repository.load_small_image(self.items[0].id)
        kept = self.repository.load_small_image(self.items[2].id)
        self.repository.register_image_items([ImageItem(self.items[0].id, ImageName("other.png"))])
        self.assertNotEqual(self.repository.load_small_image(self.items[0].id).binary, old.binary)
        self.assertIs(self.repository.load_small_image(self.items[2].id), kept)
        self.assertEqual(self.repository._thumbnail_cache_size, sum(len(image.binary) for image in self.repository._thumbnail_cache.values()))

    def test_registering_results_never_accepts_external_paths(self):
        for index, path in enumerate(("../outside.png", str(self.root / "startup.png"), "C:\\outside.png")):
            image_id = ImageId(str(index))
            self.repository.register_image_items([ImageItem(image_id, ImageName(path))])
            self.assertNotIn(image_id, self.repository._id_to_path)


if __name__ == "__main__":
    unittest.main()
