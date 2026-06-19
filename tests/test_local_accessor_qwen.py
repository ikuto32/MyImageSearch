import sys
import types
import unittest


def _install_dummy_modules():
    numpy_mod = types.ModuleType("numpy")
    numpy_mod.ndarray = object
    numpy_mod.float32 = "float32"
    numpy_mod.asarray = lambda value, dtype=None: value
    numpy_mod.load = lambda *_args, **_kwargs: None

    pil_mod = types.ModuleType("PIL")
    pil_image_mod = types.ModuleType("PIL.Image")
    class FakePilImage:
        pass

    pil_image_mod.Image = FakePilImage
    pil_image_mod.open = lambda *_args, **_kwargs: FakePilImage()
    pil_mod.Image = pil_image_mod

    pandas_mod = types.ModuleType("pandas")
    pandas_mod.read_sql_query = lambda *_args, **_kwargs: None
    pandas_mod.isna = lambda value: False

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda value: value

    faiss_mod = types.ModuleType("faiss")
    faiss_mod.read_index = lambda *_args, **_kwargs: None

    open_clip_mod = types.ModuleType("open_clip")
    open_clip_mod.create_model_and_transforms = lambda *_args, **_kwargs: (None, None, None)
    open_clip_mod.get_tokenizer = lambda *_args, **_kwargs: None

    torch_mod = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, value, device="cpu"):
            self.value = value
            self.device = device

        def to(self, device):
            return FakeTensor(self.value, device=device)

    torch_cuda = types.ModuleType("torch.cuda")
    torch_cuda.is_available = lambda: False
    torch_mod.FakeTensor = FakeTensor
    torch_mod.Tensor = FakeTensor
    torch_mod.is_tensor = lambda value: isinstance(value, FakeTensor)
    torch_mod.cuda = torch_cuda
    torch_mod.float16 = "float16"
    torch_mod.float32 = "float32"
    torch_mod.no_grad = lambda: None

    sys.modules.update(
        {
            "numpy": numpy_mod,
            "PIL": pil_mod,
            "PIL.Image": pil_image_mod,
            "pandas": pandas_mod,
            "tqdm": tqdm_mod,
            "faiss": faiss_mod,
            "open_clip": open_clip_mod,
            "torch": torch_mod,
            "torch.cuda": torch_cuda,
        }
    )


_install_dummy_modules()

from app.infrastructure.local_accessor import QwenEmbeddingBackend, torch


class FakeBatchEncoding(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_device = None

    def to(self, device):
        self.to_device = device
        return FakeBatchEncoding(
            {
                key: self._move_value(value, device)
                for key, value in self.items()
            }
        )

    @classmethod
    def _move_value(cls, value, device):
        if hasattr(value, "to"):
            return value.to(device)
        if isinstance(value, dict):
            return {key: cls._move_value(child, device) for key, child in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._move_value(child, device) for child in value)
        if isinstance(value, list):
            return [cls._move_value(child, device) for child in value]
        return value


class QwenEmbeddingBackendMoveInputsTests(unittest.TestCase):
    def _backend(self, device="cuda:0"):
        backend = QwenEmbeddingBackend.__new__(QwenEmbeddingBackend)
        backend.device = device
        return backend

    def test_uses_batch_encoding_native_to_method(self):
        backend = self._backend("cuda:0")
        inputs = FakeBatchEncoding(
            {
                "input_ids": torch.FakeTensor([1, 2]),
                "metadata": {"keep": "unchanged"},
            }
        )

        moved = backend._move_inputs_to_device(inputs)

        self.assertEqual(inputs.to_device, "cuda:0")
        self.assertIsInstance(moved, FakeBatchEncoding)
        self.assertEqual(moved["input_ids"].device, "cuda:0")
        self.assertEqual(moved["metadata"], {"keep": "unchanged"})

    def test_recursively_moves_nested_mapping_and_sequence_tensors(self):
        backend = self._backend("cuda:1")
        tensor = torch.FakeTensor
        inputs = {
            "plain_tensor": tensor("a"),
            "nested": {
                "list": [tensor("b"), "unchanged"],
                "tuple": (tensor("c"), {"inner": tensor("d")}),
            },
            "none": None,
        }

        moved = backend._move_inputs_to_device(inputs)

        self.assertEqual(moved["plain_tensor"].device, "cuda:1")
        self.assertEqual(moved["nested"]["list"][0].device, "cuda:1")
        self.assertEqual(moved["nested"]["list"][1], "unchanged")
        self.assertEqual(moved["nested"]["tuple"][0].device, "cuda:1")
        self.assertEqual(moved["nested"]["tuple"][1]["inner"].device, "cuda:1")
        self.assertIsNone(moved["none"])

    def test_moves_plain_tensor(self):
        backend = self._backend("cuda:2")
        moved = backend._move_inputs_to_device(torch.FakeTensor("value"))

        self.assertEqual(moved.device, "cuda:2")


if __name__ == "__main__":
    unittest.main()
