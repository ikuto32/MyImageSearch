import sys
import types
import unittest

# The result-finalization helpers under test do not need native ML/image deps.
# Some CI/dev shells used for lightweight tests do not install them, so provide
# import-time stubs only when the real modules are absent.
numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object
sys.modules.setdefault("numpy", numpy_stub)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
if "PIL" not in sys.modules:
    pil_module = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")
    pil_module.Image = pil_image_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image_module

from app.application.usecase import Usecase
from app.domain.domain_object import (
    ImageId,
    ImageItem,
    ImageName,
    ResultImageItem,
    Score,
)


def make_result(name: str, score: float) -> ResultImageItem:
    return ResultImageItem(
        ImageItem(ImageId(name), ImageName(name)),
        Score(score),
    )


class UsecaseResultSortingTests(unittest.TestCase):
    def setUp(self):
        self.usecase = Usecase.__new__(Usecase)

    def test_finalize_result_sorts_score_desc_then_name_asc_and_limits(self):
        results = [
            make_result("image10.png", 0.5),
            make_result("image02.png", 0.9),
            make_result("image01.png", 0.9),
            make_result("image00.png", 0.1),
        ]

        finalized = self.usecase._finalize_result(results, "query", 3)

        self.assertEqual(
            [result.item.display_name.name for result in finalized.list],
            ["image01.png", "image02.png", "image10.png"],
        )
        self.assertEqual(finalized.search_query, "query")

    def test_normalize_result_size_falls_back_to_default(self):
        self.assertEqual(self.usecase._normalize_result_size(None), 2048)
        self.assertEqual(self.usecase._normalize_result_size("invalid"), 2048)
        self.assertEqual(self.usecase._normalize_result_size(0), 1)


if __name__ == "__main__":
    unittest.main()
