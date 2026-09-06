import sys
import types
import unittest


def _install_lightweight_stubs():
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))

    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable: iterable
    sys.modules.setdefault("tqdm", tqdm_stub)

_install_lightweight_stubs()

from app.application.usecase import Usecase
from app.domain.domain_object import Image, ImageId, ImageName


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


class ZipRepository:
    def __init__(self):
        self.created_with = None

    def load_image(self, image_id: ImageId):
        if image_id.id in {"missing_image", "missing_both"}:
            raise ValueError("unknown image")
        if image_id.id == "legacy_none":
            return None
        return Image(binary=f"binary:{image_id.id}".encode(), content_type="image/png")

    def get_image_name(self, image_id: ImageId):
        if image_id.id in {"missing_name", "missing_both"}:
            raise ValueError("unknown name")
        return ImageName(f"{image_id.id}.png")

    def create_zip_from_images(self, images_with_names):
        self.created_with = images_with_names
        return "zip-buffer"


class MissingImageHandlingTests(unittest.TestCase):
    def test_get_images_zip_skips_ids_when_image_or_name_lookup_fails(self):
        repository = ZipRepository()
        usecase = Usecase.__new__(Usecase)
        usecase._repository = repository
        usecase._logger = _Logger()

        result = usecase.get_images_zip([
            "valid_1",
            "missing_image",
            "missing_name",
            "legacy_none",
            "valid_2",
        ])

        self.assertEqual(result, "zip-buffer")
        self.assertEqual(
            [(image.binary, name.name) for image, name in repository.created_with],
            [(b"binary:valid_1", "valid_1.png"), (b"binary:valid_2", "valid_2.png")],
        )


class ControllerMissingImageTests(unittest.TestCase):
    def test_unknown_small_and_original_image_return_404(self):
        from unittest.mock import patch
        from app.presentation import controller

        fake = types.SimpleNamespace(
            get_small_image=lambda _image_id: (_ for _ in ()).throw(ValueError("unknown")),
            get_image=lambda _image_id: (_ for _ in ()).throw(ValueError("unknown")),
        )
        with patch.object(controller, "usecase", fake):
            client = controller.app.test_client()
            for variant in ("small", "original"):
                with self.subTest(variant=variant):
                    response = client.get(f"/image/unknown/{variant}")
                    self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()
