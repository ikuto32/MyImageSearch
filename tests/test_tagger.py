import sys
import types

import unittest

import numpy as np
from PIL import Image


def _install_dummy_modules():
    dummy_modules = {
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
    torch_utils_data_dataset = types.ModuleType("torch.utils.data.dataset")
    torch_utils_data_utils = types.ModuleType("torch.utils.data._utils")
    torch_utils_data_utils_collate = types.ModuleType(
        "torch.utils.data._utils.collate"
    )
    torch_nn = types.ModuleType("torch.nn")
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_cuda = types.ModuleType("torch.cuda")
    torchvision_mod = types.ModuleType("torchvision")
    torchvision_transforms = types.ModuleType("torchvision.transforms")

    class _DummyLoader:  # pragma: no cover - helpers
        def __init__(self, *args, **kwargs):
            pass

    class _DummyDataset:
        pass

    class _DummyModule:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            return args[0] if args else None

        def to(self, *_, **__):
            return self

        def eval(self):
            return self

        def load_state_dict(self, *_, **__):
            return None

    class _Linear(_DummyModule):
        pass

    class _SiLU(_DummyModule):
        pass

    class _Dropout(_DummyModule):
        pass

    class _ReLU(_DummyModule):
        pass

    def _sequential(*modules):
        seq = _DummyModule()
        seq.modules = modules
        return seq

    def _no_grad(fn=None):
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

            def __call__(self, func):
                return func

        if fn is None:
            return _NoGrad()

        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    torch_nn.Module = _DummyModule
    torch_nn.Sequential = _sequential
    torch_nn.Linear = _Linear
    torch_nn.SiLU = _SiLU
    torch_nn.Dropout1d = _Dropout
    torch_nn.Dropout = _Dropout
    torch_nn.ReLU = _ReLU

    def _default_collate(batch):
        return batch

    torch_utils_data.DataLoader = _DummyLoader
    torch_utils_data.Dataset = _DummyDataset
    torch_utils_data.IterableDataset = _DummyDataset
    torch_utils_data.get_worker_info = lambda: None
    torch_utils_data_dataset = torch_utils_data_dataset
    torch_utils_data_utils.collate = torch_utils_data_utils_collate
    torch_utils_data_utils_collate.default_collate = _default_collate

    torch_mod.utils = torch_utils
    torch_mod.utils.data = torch_utils_data
    torch_mod.nn = torch_nn
    torch_mod.cuda = torch_cuda
    torch_mod.no_grad = _no_grad
    torch_mod.load = lambda *_, **__: {}
    torch_mod.cdist = lambda x, y: None
    torch_mod.norm = lambda x, dim=None, keepdim=None: x
    torch_mod.tensor = lambda x, **__: x
    torch_mod.from_numpy = lambda x: x

    torch_cuda.is_available = lambda: False

    torchvision_mod.transforms = torchvision_transforms

    dummy_modules.update(
        {
            "torch": torch_mod,
            "torch.utils": torch_utils,
            "torch.utils.data": torch_utils_data,
            "torch.utils.data.dataset": torch_utils_data_dataset,
            "torch.utils.data._utils": torch_utils_data_utils,
            "torch.utils.data._utils.collate": torch_utils_data_utils_collate,
            "torch.nn": torch_nn,
            "torch.nn.functional": torch_nn_functional,
            "torch.cuda": torch_cuda,
            "torchvision": torchvision_mod,
            "torchvision.transforms": torchvision_transforms,
        }
    )

    dummy_modules["onnxruntime"].InferenceSession = object

    sys.modules.update(dummy_modules)


_install_dummy_modules()

from create_index import prepare_tag_input


class TagPreprocessingTests(unittest.TestCase):
    def test_prepare_tag_input_consistency(self):
        target_size = 16
        images = [
            Image.new("RGB", (8, 12), color="red"),
            Image.new("RGB", (20, 10), color="blue"),
        ]

        prepared = [prepare_tag_input(img, target_size) for img in images]
        stacked = np.stack(prepared)

        self.assertEqual(
            stacked.shape,
            (
                len(images),
                target_size,
                target_size,
                3,
            ),
        )
        for single, batch in zip(prepared, stacked):
            np.testing.assert_array_equal(single, batch)

    def test_prepare_tag_input_bgr_and_padding(self):
        target_size = 8
        image = Image.new("RGB", (4, 6), color=(10, 20, 30))

        arr = prepare_tag_input(image, target_size)

        self.assertEqual(arr.shape, (target_size, target_size, 3))
        self.assertEqual(arr.dtype, np.float32)
        center = arr[target_size // 2, target_size // 2]
        np.testing.assert_array_equal(center, np.array([30, 20, 10], dtype=np.float32))
        np.testing.assert_array_equal(arr[0, 0], np.array([255.0, 255.0, 255.0]))


if __name__ == "__main__":
    unittest.main()
