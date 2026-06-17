import json
import unittest

from app.domain.domain_object import ImageId, ModelId
from app.presentation import controller


QWEN_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_PRETRAINED = ""


class FakeUsecase:
    def __init__(self):
        self.metadata_calls = []
        self.rating_calls = []

    def get_image_metadata(self, model_id: ModelId, image_id: ImageId):
        self.metadata_calls.append((model_id, image_id))
        return {
            "tags": "qwen tag",
            "style_cluster": "cluster-a",
            "rating": "safe",
            "aesthetic_quality": 7.5,
        }

    def get_rating_list(self, model_id: ModelId, image_ids: list[ImageId] | None = None):
        self.rating_calls.append((model_id, image_ids))
        ids = image_ids or [ImageId("fallback")]
        return {image_id: f"rating-{image_id.id}" for image_id in ids}


class ControllerModelParamTests(unittest.TestCase):
    def setUp(self):
        self.usecase = FakeUsecase()
        controller.usecase = self.usecase
        self.client = controller.app.test_client()

    def test_image_meta_accepts_qwen_model_id_from_query_string(self):
        response = self.client.get(
            "/image_meta/image-001",
            query_string={
                "model_name": QWEN_MODEL_NAME,
                "pretrained": QWEN_PRETRAINED,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data), {
            "tags": "qwen tag",
            "style_cluster": "cluster-a",
            "rating": "safe",
            "aesthetic_quality": 7.5,
        })
        self.assertEqual(
            self.usecase.metadata_calls,
            [(ModelId(QWEN_MODEL_NAME, QWEN_PRETRAINED), ImageId("image-001"))],
        )

    def test_image_ratings_accepts_qwen_model_id_and_ids_from_post_params(self):
        response = self.client.post(
            "/image_ratings",
            json={
                "params": {
                    "model_name": QWEN_MODEL_NAME,
                    "pretrained": QWEN_PRETRAINED,
                    "ids": ["image-001", "image-002"],
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data), {
            "image-001": "rating-image-001",
            "image-002": "rating-image-002",
        })
        self.assertEqual(
            self.usecase.rating_calls,
            [(
                ModelId(QWEN_MODEL_NAME, QWEN_PRETRAINED),
                [ImageId("image-001"), ImageId("image-002")],
            )],
        )


if __name__ == "__main__":
    unittest.main()
