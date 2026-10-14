import base64
import io
import threading
import types
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch

from indexer.embedding import QwenVlEmbeddingBackend, normalize_qwen_api_base, pil_image_to_data_url


def response(embedding):
    return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=embedding)])


def backend(output_dim=2, max_concurrency=2) -> QwenVlEmbeddingBackend:
    """A backend that never contacts a server unless a test replaces its client."""
    instance = QwenVlEmbeddingBackend.__new__(QwenVlEmbeddingBackend)
    instance.model_id = "fake-qwen"
    instance.output_dim = output_dim
    instance.api_base = "http://127.0.0.1:8000/v1"
    instance.api_key = "EMPTY"
    instance.timeout = 300.0
    instance.max_retries = 2
    instance.max_concurrency = max_concurrency
    instance.instruction = "Represent the user's input."
    instance._thread_local = threading.local()
    return instance


class QwenApiBackendTests(unittest.TestCase):
    def test_api_base_normalization(self):
        self.assertEqual(normalize_qwen_api_base("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1")
        self.assertEqual(normalize_qwen_api_base("http://127.0.0.1:8000/v1"), "http://127.0.0.1:8000/v1")
        self.assertEqual(normalize_qwen_api_base(" https://host/v1/ "), "https://host/v1")
        for invalid in ("", "  ", None, "127.0.0.1:8000", "ftp://host/v1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_qwen_api_base(invalid)

    def test_constructor_validates_options_before_contacting_the_server(self):
        for options in (
            {"timeout": 0}, {"max_retries": -1}, {"max_concurrency": 0}, {"instruction": ""}, {"api_base": "host"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                QwenVlEmbeddingBackend("fake-qwen", skip_server_check=True, **options)

    def test_images_are_sent_as_rgb_png_data_urls(self):
        data_url = pil_image_to_data_url(Image.new("L", (4, 5), 128))
        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as decoded:
            self.assertEqual((decoded.format, decoded.mode, decoded.size), ("PNG", "RGB", (4, 5)))

    def test_image_messages_and_body(self):
        instance = backend()
        messages = instance._image_messages("data:image/png;base64,abc")
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[0]["content"][0]["text"], "Represent the user's input.")
        self.assertEqual(messages[1]["content"][0]["image_url"]["url"], "data:image/png;base64,abc")
        self.assertEqual(messages[1]["content"][1]["text"], "")
        self.assertEqual(messages[2]["content"][0]["text"], "")
        self.assertEqual(instance._embedding_body(messages), {
            "model": "fake-qwen",
            "messages": messages,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        })

    def test_batch_encoding_keeps_input_order_and_truncates(self):
        instance = backend(output_dim=2, max_concurrency=3)
        embeddings = [[1, 2, 99], [3, 4, 99], [5, 6, 99]]
        release = threading.Barrier(3)

        def post(_path, *, cast_to, body):
            # Every request is in flight at once; results must still follow input order.
            release.wait(timeout=5)
            width = Image.open(io.BytesIO(base64.b64decode(
                body["messages"][1]["content"][0]["image_url"]["url"].split(",", 1)[1]
            ))).width
            return response(embeddings[width - 1])

        instance._get_client = lambda: types.SimpleNamespace(post=post)
        images = [Image.new("RGB", (width, 1)) for width in (1, 2, 3)]
        internal, features = instance.encode_image_with_internal(images)
        self.assertIs(internal, features)
        self.assertEqual((features.dtype, features.device.type), (torch.float32, "cpu"))
        np.testing.assert_array_equal(features.numpy(), [[1, 2], [3, 4], [5, 6]])

    def test_only_nonempty_lists_of_images_are_accepted(self):
        for invalid in ([], ["not-image"], Image.new("RGB", (1, 1)), torch.zeros(1, 3, 2, 2)):
            with self.subTest(invalid=type(invalid).__name__), self.assertRaises(TypeError):
                backend().encode_image_with_internal(invalid)

    def test_response_validation_errors(self):
        instance = backend(output_dim=3)
        for data, message in (
            (types.SimpleNamespace(data=[]), "any data"),
            (types.SimpleNamespace(data=[types.SimpleNamespace(embedding=None)]), "embedding"),
            (response([]), "empty"),
            (response([[1, 2, 3]]), "1-dimensional"),
            (response([1, 2]), "smaller"),
            (response([1, float("nan"), 3]), "NaN or Inf"),
            (response([1, float("inf"), 3]), "NaN or Inf"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                instance._validate_embedding_response(data, index=0)
        self.assertEqual(backend(output_dim=None)._validate_embedding_response(response([1, 2, 3])).tolist(), [1, 2, 3])

    def test_server_check_requires_the_served_model(self):
        instance = backend()
        models = Mock()
        models.list.return_value = types.SimpleNamespace(data=[types.SimpleNamespace(id="other-model")])
        instance._get_client = lambda: types.SimpleNamespace(models=models)
        with self.assertRaisesRegex(RuntimeError, "fake-qwen.*Available models: other-model"):
            instance._check_server()
        models.list.return_value = types.SimpleNamespace(data=[types.SimpleNamespace(id="fake-qwen")])
        instance._check_server()


if __name__ == "__main__":
    unittest.main()
