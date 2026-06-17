import importlib.util
import sys
import types
import unittest
from pathlib import Path

# The controller import does not need native ML/image deps for these route tests.
if importlib.util.find_spec("numpy") is None:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object
    numpy_stub.set_printoptions = lambda *args, **kwargs: None
    sys.modules.setdefault("numpy", numpy_stub)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
if importlib.util.find_spec("PIL") is None and "PIL" not in sys.modules:
    pil_module = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")
    pil_image_module.Image = object
    pil_module.Image = pil_image_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image_module

from app.presentation import controller


class ResourceRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = controller.app.test_client()
        self.presentation_dir = Path(controller.__file__).parent
        self.evil_dir = self.presentation_dir / "view_evil"
        self.evil_file = self.evil_dir / "file.js"
        self.evil_dir.mkdir(exist_ok=True)
        self.evil_file.write_text("alert('evil');", encoding="UTF-8")

    def tearDown(self):
        self.evil_file.unlink(missing_ok=True)
        self.evil_dir.rmdir()

    def test_rejects_sibling_view_evil_path_traversal(self):
        response = self.client.get("/../view_evil/file.js")

        self.assertEqual(response.status_code, 403)

    def test_rejects_url_encoded_dot_dot_path_traversal(self):
        response = self.client.get("/js/%2e%2e/%2e%2e/view_evil/file.js")

        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
