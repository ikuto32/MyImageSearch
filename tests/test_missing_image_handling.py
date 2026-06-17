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


def _install_controller_stubs():
    class AbortException(Exception):
        def __init__(self, code):
            self.code = code
            super().__init__(code)

    class FakeFlask:
        def __init__(self, *_args, **_kwargs):
            pass

        def route(self, *_args, **_kwargs):
            return lambda func: func

    class Headers(dict):
        def set(self, key, value):
            self[key] = value

    class Response:
        def __init__(self, binary):
            self.binary = binary
            self.headers = Headers()

    flask_stub = types.ModuleType("flask")
    flask_stub.Flask = FakeFlask
    flask_stub.request = types.SimpleNamespace(args={}, form={}, files={})
    flask_stub.make_response = Response
    flask_stub.send_file = lambda *args, **kwargs: (args, kwargs)
    flask_stub.abort = lambda code: (_ for _ in ()).throw(AbortException(code))
    sys.modules["flask"] = flask_stub

    logging_config_stub = types.ModuleType("app.logging_config")
    logging_config_stub.configure_logging = lambda: None
    sys.modules["app.logging_config"] = logging_config_stub
    return AbortException


class ControllerMissingImageTests(unittest.TestCase):
    def test_unknown_small_and_original_image_return_404(self):
        AbortException = _install_controller_stubs()
        sys.modules.pop("app.presentation.controller", None)
        from app.presentation import controller

        controller.usecase = types.SimpleNamespace(
            get_small_image=lambda _image_id: (_ for _ in ()).throw(ValueError("unknown")),
            get_image=lambda _image_id: (_ for _ in ()).throw(ValueError("unknown")),
        )

        with self.assertRaises(AbortException) as small_error:
            controller.get_small_image("unknown")
        self.assertEqual(small_error.exception.code, 404)

        with self.assertRaises(AbortException) as original_error:
            controller.get_original_image("unknown")
        self.assertEqual(original_error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
