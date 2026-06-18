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
                "attention_mask": tensor([1, 1]),
                "pixel_values": tensor([[1, 2], [3, 4]]),
                "image_grid_thw": tensor([[1, 1, 2]]),
            }, None, 0, tensor([1]), tensor([2])),
            ({
                "input_ids": tensor([201, 202]),
                "attention_mask": tensor([1, 1]),
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

    def test_prepare_image_inputs_adds_prompt_tokens_and_images(self):
        tensor = create_index.torch.FakeTensor
        images = [create_index.Image.Image(), create_index.Image.Image()]
        calls = {}

        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                calls["messages"] = messages
                calls["tokenize"] = tokenize
                calls["add_generation_prompt"] = add_generation_prompt
                return "<chat><image>Represent this image for retrieval."

            def __call__(self, **kwargs):
                calls["processor_kwargs"] = kwargs
                return {
                    "input_ids": tensor([[1, 2], [1, 2]]),
                    "attention_mask": tensor([[1, 1], [1, 1]]),
                    "pixel_values": tensor([[3, 4], [5, 6]]),
                    "image_grid_thw": tensor([[1, 1, 2], [1, 1, 2]]),
                }

        backend = QwenVlEmbeddingBackend.__new__(QwenVlEmbeddingBackend)
        backend.processor = FakeProcessor()

        inputs = backend.prepare_image_inputs(images)

        self.assertIn("input_ids", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertEqual(calls["processor_kwargs"]["images"], images)
        self.assertEqual(calls["processor_kwargs"]["text"], ["<chat><image>Represent this image for retrieval."] * 2)
        self.assertFalse(calls["tokenize"])
        self.assertFalse(calls["add_generation_prompt"])


    def test_qwen_backend_configures_processor_max_pixels(self):
        calls = {}

        class FakeImageProcessor:
            def __init__(self):
                self.min_pixels = 3136
                self.max_pixels = None
                self.size = {"longest_edge": 999999, "shortest_edge": 28}

        class FakeProcessor:
            def __init__(self):
                self.image_processor = FakeImageProcessor()

        fake_processor = FakeProcessor()

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(model_id, trust_remote_code=True):
                calls["processor_model_id"] = model_id
                calls["processor_trust_remote_code"] = trust_remote_code
                return fake_processor

        class FakeParameter:
            device = "cpu"
            dtype = "float32"

        class FakeConfig:
            _attn_implementation = None
            use_cache = None
            text_config = None
            vision_config = None

        class FakeModel:
            config = FakeConfig()

            def to(self, device):
                calls["device"] = device
                return self

            def eval(self):
                calls["eval"] = True
                return self

            def parameters(self):
                return iter([FakeParameter()])

        class FakeQwen3VLModel:
            @staticmethod
            def from_pretrained(model_id, torch_dtype=None, trust_remote_code=True):
                calls["model_id"] = model_id
                calls["torch_dtype"] = torch_dtype
                calls["model_trust_remote_code"] = trust_remote_code
                return FakeModel()

        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoProcessor = FakeAutoProcessor
        transformers_mod.Qwen3VLModel = FakeQwen3VLModel
        old_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = transformers_mod
        try:
            backend = QwenVlEmbeddingBackend("fake-qwen", "cpu", output_dim=128, max_pixels=123456)
        finally:
            if old_transformers is None:
                del sys.modules["transformers"]
            else:
                sys.modules["transformers"] = old_transformers

        self.assertIs(backend.processor, fake_processor)
        self.assertEqual(fake_processor.image_processor.max_pixels, 123456)
        self.assertEqual(fake_processor.image_processor.size["longest_edge"], 123456)
        self.assertEqual(fake_processor.image_processor.size["shortest_edge"], 28)
        self.assertEqual(calls["processor_model_id"], "fake-qwen")
        self.assertEqual(calls["model_id"], "fake-qwen")

    def test_qwen_max_pixels_cli_default_custom_and_validation(self):
        original_argv = sys.argv[:]
        try:
            sys.argv = ["create_index.py"]
            args = create_index.parse_arguments()
            self.assertEqual(args.qwen_max_pixels, 262144)

            sys.argv = ["create_index.py", "--qwen-max-pixels", "131072"]
            args = create_index.parse_arguments()
            self.assertEqual(args.qwen_max_pixels, 131072)

            sys.argv = ["create_index.py", "--qwen-max-pixels", "0"]
            with self.assertRaises(SystemExit):
                create_index.parse_arguments()
        finally:
            sys.argv = original_argv

    def test_load_search_embedding_backend_passes_qwen_max_pixels_only_to_qwen(self):
        original_qwen = create_index.QwenVlEmbeddingBackend
        original_openclip = create_index.OpenClipEmbeddingBackend
        seen = {}

        class FakeQwenBackend:
            def __init__(self, model_id, device, output_dim=None, max_pixels=262144):
                seen["qwen"] = (model_id, device, output_dim, max_pixels)

        class FakeOpenClipBackend:
            def __init__(self, model_name, pretrained, device):
                seen["open_clip"] = (model_name, pretrained, device)

        create_index.QwenVlEmbeddingBackend = FakeQwenBackend
        create_index.OpenClipEmbeddingBackend = FakeOpenClipBackend
        try:
            qwen_args = types.SimpleNamespace(
                search_backend="qwen_vl",
                search_model_id="fake-qwen",
                search_model_out_dim=64,
                qwen_max_pixels=777,
            )
            create_index.load_search_embedding_backend(qwen_args, "cpu")
            self.assertEqual(seen["qwen"], ("fake-qwen", "cpu", 64, 777))

            open_clip_args = types.SimpleNamespace(
                search_backend="open_clip",
                search_model_name="ViT-L-14",
                search_model_pretrained="openai",
            )
            create_index.load_search_embedding_backend(open_clip_args, "cpu")
            self.assertEqual(seen["open_clip"], ("ViT-L-14", "openai", "cpu"))
        finally:
            create_index.QwenVlEmbeddingBackend = original_qwen
            create_index.OpenClipEmbeddingBackend = original_openclip

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
