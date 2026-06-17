import sys
import types
import unittest



def _install_dummy_modules():
    numpy_mod = types.ModuleType("numpy")
    numpy_mod.ndarray = object
    numpy_mod.float32 = "float32"

    pil_mod = types.ModuleType("PIL")
    pil_image_mod = types.ModuleType("PIL.Image")
    pil_image_file_mod = types.ModuleType("PIL.ImageFile")
    class FakePilImage:
        pass

    pil_image_mod.MAX_IMAGE_PIXELS = None
    pil_image_mod.Image = FakePilImage
    pil_image_mod.LANCZOS = 1
    pil_mod.Image = pil_image_mod
    pil_mod.ImageFile = pil_image_file_mod

    dummy_modules = {
        "numpy": numpy_mod,
        "PIL": pil_mod,
        "PIL.Image": pil_image_mod,
        "PIL.ImageFile": pil_image_file_mod,
        "faiss": types.ModuleType("faiss"),
        "huggingface_hub": types.ModuleType("huggingface_hub"),
        "open_clip": types.ModuleType("open_clip"),
        "pandas": types.ModuleType("pandas"),
        "tqdm": types.ModuleType("tqdm"),
        "onnxruntime": types.ModuleType("onnxruntime"),
    }

    torch_mod = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data_utils = types.ModuleType("torch.utils.data._utils")
    torch_utils_data_utils_collate = types.ModuleType("torch.utils.data._utils.collate")
    torch_nn = types.ModuleType("torch.nn")
    torch_cuda = types.ModuleType("torch.cuda")
    torchvision_mod = types.ModuleType("torchvision")
    torchvision_transforms = types.ModuleType("torchvision.transforms")

    class FakeTensor:
        def __init__(self, data):
            self.data = data
            self._shape = self._infer_shape(data)

        @property
        def shape(self):
            return self._shape

        @property
        def ndim(self):
            return len(self._shape)

        @property
        def dtype(self):
            return type(self._first_scalar(self.data))

        @property
        def device(self):
            return "cpu"

        def new_full(self, shape, fill_value):
            return FakeTensor(self._filled(shape, fill_value))

        def __setitem__(self, key, value):
            self._assign(self.data, key, value.data)

        def __getitem__(self, key):
            return FakeTensor(self.data[key])

        def to(self, *_, **__):
            return self

        def squeeze(self, dim=None):
            if dim is None:
                return FakeTensor(self._squeeze_all(self.data))
            if dim < 0:
                dim += len(self._shape)
            if self._shape[dim] != 1:
                return self
            return FakeTensor(self._squeeze_dim(self.data, dim))

        @classmethod
        def _infer_shape(cls, data):
            if isinstance(data, list):
                return (len(data),) + (cls._infer_shape(data[0]) if data else ())
            return ()

        @classmethod
        def _squeeze_all(cls, data):
            while isinstance(data, list) and len(data) == 1:
                data = data[0]
            if isinstance(data, list):
                return [cls._squeeze_all(item) for item in data]
            return data

        @classmethod
        def _squeeze_dim(cls, data, dim):
            if dim == 0:
                return data[0]
            return [cls._squeeze_dim(item, dim - 1) for item in data]

        @classmethod
        def _filled(cls, shape, fill_value):
            if not shape:
                return fill_value
            return [cls._filled(shape[1:], fill_value) for _ in range(shape[0])]

        @classmethod
        def _first_scalar(cls, data):
            return cls._first_scalar(data[0]) if isinstance(data, list) else data

        @classmethod
        def _assign(cls, target, slices, source):
            if len(slices) == 1:
                for i, value in enumerate(source):
                    target[i] = value
                return
            for i, child in enumerate(source):
                cls._assign(target[i], slices[1:], child)

    def fake_stack(values, dim=0):
        assert dim == 0
        return FakeTensor([value.data for value in values])

    def fake_default_collate(batch):
        first = batch[0]
        if isinstance(first, FakePilImage):
            raise AssertionError("PIL images should bypass default_collate")
        if isinstance(first, FakeTensor):
            shapes = [item.shape for item in batch]
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise RuntimeError("variable tensor shapes cannot be default-collated")
            return fake_stack(batch, dim=0)
        if isinstance(first, dict):
            return {key: fake_default_collate([item[key] for item in batch]) for key in first}
        return list(batch)

    class _DummyModule:
        def __init__(self, *_, **__):
            pass

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **__):
            return args[0] if args else None

        def to(self, *_, **__):
            return self

        def eval(self):
            return self

    torch_mod.FakeTensor = FakeTensor
    torch_mod.is_tensor = lambda value: isinstance(value, FakeTensor)
    torch_mod.stack = fake_stack
    torch_mod.no_grad = lambda fn=None: (fn if fn is not None else (lambda func: func))
    torch_mod.inference_mode = torch_mod.no_grad
    torch_mod.float16 = "float16"
    torch_mod.float32 = "float32"
    torch_mod.Tensor = FakeTensor
    torch_nn.Module = _DummyModule
    for name in ("Sequential", "Linear", "SiLU", "Dropout1d", "Dropout", "ReLU"):
        setattr(torch_nn, name, _DummyModule)
    torch_cuda.is_available = lambda: False
    torch_utils_data.DataLoader = object
    torch_utils_data.IterableDataset = object
    torch_utils_data.get_worker_info = lambda: None
    torch_utils_data_utils.collate = torch_utils_data_utils_collate
    torch_utils_data_utils_collate.default_collate = fake_default_collate
    torchvision_mod.transforms = torchvision_transforms

    dummy_modules.update({
        "torch": torch_mod,
        "torch.nn": torch_nn,
        "torch.cuda": torch_cuda,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torch.utils.data._utils": torch_utils_data_utils,
        "torch.utils.data._utils.collate": torch_utils_data_utils_collate,
        "torchvision": torchvision_mod,
        "torchvision.transforms": torchvision_transforms,
    })
    dummy_modules["onnxruntime"].InferenceSession = object
    sys.modules.update(dummy_modules)


_install_dummy_modules()

import create_index
from create_index import QwenVlEmbeddingBackend, safe_collate


class QwenCollateTests(unittest.TestCase):
    def test_qwen_processor_dicts_are_padded_before_encoding(self):
        tensor = create_index.torch.FakeTensor
        samples = [
            ({
                "input_ids": tensor([101, 102]),
                "pixel_values": tensor([[1, 2], [3, 4]]),
                "image_grid_thw": tensor([[1, 1, 2]]),
            }, None, 0, tensor([1]), tensor([2])),
            ({
                "input_ids": tensor([201, 202]),
                "pixel_values": tensor([[5, 6], [7, 8], [9, 10]]),
                "image_grid_thw": tensor([[1, 1, 3]]),
            }, None, 1, tensor([3]), tensor([4])),
        ]

        search_batch, _, _, _, _ = safe_collate(samples)

        self.assertEqual(search_batch["input_ids"].shape, (2, 2))
        self.assertEqual(search_batch["pixel_values"].shape, (2, 3, 2))
        self.assertEqual(search_batch["pixel_values"].data[0][2], [0, 0])

        backend = QwenVlEmbeddingBackend.__new__(QwenVlEmbeddingBackend)
        backend.device = "cpu"
        backend.output_dim = None
        seen = {}

        class FakeModel:
            def __call__(self, **kwargs):
                provided_embedding_inputs = sum(
                    key in kwargs and kwargs[key] is not None
                    for key in ("input_ids", "inputs_embeds")
                )
                if provided_embedding_inputs != 1:
                    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
                seen.update(kwargs)
                return types.SimpleNamespace(pooler_output=tensor([[1.0], [2.0]]))

        backend.model = FakeModel()
        internal, features = backend.encode_image_with_internal(search_batch)

        self.assertIs(internal, features)
        self.assertEqual(features.shape, (2, 1))
        self.assertEqual(seen["input_ids"].shape, (2, 2))
        self.assertEqual(seen["pixel_values"].shape, (2, 3, 2))
        self.assertEqual(seen["image_grid_thw"].shape, (2, 3))


    def test_pil_image_inputs_bypass_default_collate(self):
        tensor = create_index.torch.FakeTensor
        images = [create_index.Image.Image(), create_index.Image.Image()]
        samples = [
            (images[0], None, 0, tensor([1]), tensor([2])),
            (images[1], None, 1, tensor([3]), tensor([4])),
        ]

        search_batch, _, _, _, _ = safe_collate(samples)

        self.assertIsInstance(search_batch, list)
        self.assertEqual(search_batch, images)
        self.assertTrue(all(isinstance(item, create_index.Image.Image) for item in search_batch))

    def test_openclip_tensor_inputs_stay_on_default_collate_path(self):
        tensor = create_index.torch.FakeTensor
        samples = [
            (tensor([[1, 2], [3, 4]]), None, 0, tensor([1]), tensor([2])),
            (tensor([[5, 6], [7, 8]]), None, 1, tensor([3]), tensor([4])),
        ]

        search_batch, _, _, _, _ = safe_collate(samples)

        self.assertEqual(search_batch.shape, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
