import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from app.domain.domain_object import ModelId
from app.infrastructure import embedding_backends
from app.infrastructure.embedding_backends import QwenEmbeddingBackend, create_embedding_backend


class FakeTensor:
    def __init__(self, value, device="cpu"):
        self.value = value
        self.device = device

    def to(self, device):
        return FakeTensor(self.value, device=device)


class FakeBatchEncoding(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_device = None

    def to(self, device):
        self.to_device = device
        return FakeBatchEncoding({
            key: value.to(device) if hasattr(value, "to") else value for key, value in self.items()
        })


def backend(device="cuda:0") -> QwenEmbeddingBackend:
    instance = QwenEmbeddingBackend.__new__(QwenEmbeddingBackend)
    instance.device = device
    return instance


class MoveInputsTests(unittest.TestCase):
    def test_uses_batch_encoding_native_to_method(self):
        inputs = FakeBatchEncoding({"input_ids": FakeTensor([1, 2]), "metadata": {"keep": "unchanged"}})

        moved = backend("cuda:0")._move_inputs_to_device(inputs)

        self.assertEqual(inputs.to_device, "cuda:0")
        self.assertIsInstance(moved, FakeBatchEncoding)
        self.assertEqual(moved["input_ids"].device, "cuda:0")
        self.assertEqual(moved["metadata"], {"keep": "unchanged"})

    def test_recursively_moves_nested_mapping_and_sequence_tensors(self):
        inputs = {
            "plain_tensor": FakeTensor("a"),
            "nested": {
                "list": [FakeTensor("b"), "unchanged"],
                "tuple": (FakeTensor("c"), {"inner": FakeTensor("d")}),
            },
            "none": None,
        }

        moved = backend("cuda:1")._move_inputs_to_device(inputs)

        self.assertEqual(moved["plain_tensor"].device, "cuda:1")
        self.assertEqual(moved["nested"]["list"][0].device, "cuda:1")
        self.assertEqual(moved["nested"]["list"][1], "unchanged")
        self.assertEqual(moved["nested"]["tuple"][0].device, "cuda:1")
        self.assertEqual(moved["nested"]["tuple"][1]["inner"].device, "cuda:1")
        self.assertIsNone(moved["none"])


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.backend = backend("cpu")
        self.backend.processor = Mock()
        self.backend.processor.apply_chat_template.return_value = "<prompt>"
        self.backend.processor.return_value = {"input_ids": torch.zeros((1, 2))}
        self.backend.model = Mock(return_value=types.SimpleNamespace(
            pooler_output=None, image_embeds=None, text_embeds=None,
            last_hidden_state=torch.tensor([[[1., 3.], [3., 5.]]]),
        ))

    def test_text_and_images_are_formatted_as_a_user_turn(self):
        np.testing.assert_array_equal(self.backend.encode_text("cat"), np.array([[2., 4.]], dtype=np.float32))
        message = self.backend.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(message, [{"role": "user", "content": [{"type": "text", "text": "cat"}]}])
        self.backend.processor.assert_called_once_with(text=["<prompt>"], padding=True, return_tensors="pt")

        image = object()
        self.backend.processor.reset_mock()
        self.backend.encode_image(image)
        self.backend.processor.assert_called_once_with(
            text=["<prompt>"], padding=True, return_tensors="pt", images=[image],
        )

    def test_processors_without_a_usable_chat_template_receive_plain_inputs(self):
        self.backend.processor.apply_chat_template.side_effect = ValueError("no template")
        self.backend.encode_text("cat")
        self.backend.processor.assert_called_once_with(return_tensors="pt", text=["cat"], padding=True)
        image = object()
        self.backend.processor.reset_mock()
        self.backend.encode_image(image)
        self.backend.processor.assert_called_once_with(return_tensors="pt", images=image)

    def test_explicit_embedding_outputs_take_precedence_over_pooling(self):
        self.backend.model.return_value.text_embeds = torch.tensor([[7., 8.]])
        np.testing.assert_array_equal(self.backend.encode_text("cat"), np.array([[7., 8.]], dtype=np.float32))
        self.backend.model.return_value = types.SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "did not return embeddings"):
            self.backend.encode_text("cat")


class BackendSelectionTests(unittest.TestCase):
    def test_qwen_models_use_transformers_and_other_models_use_open_clip(self):
        with (
            patch.object(embedding_backends, "QwenEmbeddingBackend") as qwen,
            patch.object(embedding_backends, "OpenClipEmbeddingBackend") as open_clip,
        ):
            self.assertIs(create_embedding_backend(ModelId("Qwen/Qwen3-VL-Embedding-2B", "")), qwen.return_value)
            qwen.assert_called_once_with("Qwen/Qwen3-VL-Embedding-2B")
            self.assertIs(create_embedding_backend(ModelId("ViT-L-14", "openai")), open_clip.return_value)
            open_clip.assert_called_once_with("ViT-L-14", "openai")

    def test_qwen_loads_native_transformers_classes_without_remote_code(self):
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = Mock()
        transformers.Qwen3VLModel = Mock()
        with patch.dict("sys.modules", {"transformers": transformers}):
            QwenEmbeddingBackend("Qwen/test")
        transformers.AutoProcessor.from_pretrained.assert_called_once_with("Qwen/test", trust_remote_code=False)
        self.assertIs(transformers.Qwen3VLModel.from_pretrained.call_args.kwargs["trust_remote_code"], False)


if __name__ == "__main__":
    unittest.main()
