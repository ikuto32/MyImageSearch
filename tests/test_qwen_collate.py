import base64
import io
import sys
import types
import unittest



def _install_dummy_modules():
    numpy_mod = types.ModuleType("numpy")
    numpy_mod.ndarray = object
    numpy_mod.float32 = "float32"
    class FakeArray:
        def __init__(self, data):
            self.data = list(data)
            self.shape = (len(self.data),)
            self.ndim = 1
            self.size = len(self.data)
        def __getitem__(self, key):
            return FakeArray(self.data[key]) if isinstance(key, slice) else self.data[key]
        def astype(self, *_, **__):
            return self
    numpy_mod.asarray = lambda value, dtype=None: FakeArray(value)
    numpy_mod.stack = lambda values: FakeStackedArray([value.data for value in values])
    numpy_mod.isfinite = lambda array: types.SimpleNamespace(all=lambda: all(x == x and x not in (float("inf"), float("-inf")) for x in array.data))

    pil_mod = types.ModuleType("PIL")
    pil_image_mod = types.ModuleType("PIL.Image")
    pil_image_file_mod = types.ModuleType("PIL.ImageFile")
    class FakePilImage:
        def __init__(self, width=2, height=3, mode="RGB"):
            self.size = (width, height)
            self.mode = mode
        def copy(self):
            return FakePilImage(*self.size, mode=self.mode)
        def convert(self, mode):
            return FakePilImage(*self.size, mode=mode)
        def save(self, buffer, format=None):
            buffer.write(b"\x89PNG\r\n\x1a\n" + f"{self.size[0]}x{self.size[1]}:{self.mode}".encode("ascii"))
    class FakeStackedArray:
        def __init__(self, data):
            self.data = data
            self.shape = (len(data), len(data[0]) if data else 0)
        def astype(self, *_, **__):
            return self

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
        "openai": types.ModuleType("openai"),
        "openai.types": types.ModuleType("openai.types"),
        "openai.types.create_embedding_response": types.ModuleType("openai.types.create_embedding_response"),
    }
    class FakeOpenAI:
        def __init__(self, *_, **__):
            pass
    for exc_name in ("APIConnectionError", "APITimeoutError", "APIStatusError"):
        setattr(dummy_modules["openai"], exc_name, type(exc_name, (Exception,), {}))
    dummy_modules["openai"].OpenAI = FakeOpenAI
    dummy_modules["openai.types.create_embedding_response"].CreateEmbeddingResponse = object

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
            if hasattr(data, "data"):
                data = data.data
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

    def fake_empty(shape, dtype=None):
        rows, cols = shape
        return FakeTensor([[0.0 for _ in range(cols)] for _ in range(rows)])
    torch_mod.FakeTensor = FakeTensor
    torch_mod.is_tensor = lambda value: isinstance(value, FakeTensor)
    torch_mod.stack = fake_stack
    torch_mod.empty = fake_empty
    torch_mod.from_numpy = lambda value: FakeTensor(value.data)
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


class QwenApiBackendTests(unittest.TestCase):
    def _backend(self, output_dim=2):
        backend = QwenVlEmbeddingBackend.__new__(QwenVlEmbeddingBackend)
        backend.model_id = "fake-qwen"
        backend.output_dim = output_dim
        backend.api_base = "http://127.0.0.1:8000/v1"
        backend.api_key = "EMPTY"
        backend.timeout = 300.0
        backend.max_retries = 2
        backend.max_concurrency = 2
        backend.instruction = "Represent the user's input."
        backend._thread_local = types.SimpleNamespace()
        return backend

    def test_api_base_normalization(self):
        self.assertEqual(create_index.normalize_qwen_api_base("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1")
        self.assertEqual(create_index.normalize_qwen_api_base("http://127.0.0.1:8000/v1"), "http://127.0.0.1:8000/v1")
        self.assertEqual(create_index.normalize_qwen_api_base("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1")
        with self.assertRaises(ValueError):
            create_index.normalize_qwen_api_base("")

    def test_data_url_shape_and_png(self):
        image = create_index.Image.Image(4, 5, mode="L")
        data_url = create_index.pil_image_to_data_url(image)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        decoded = base64.b64decode(data_url.split(",", 1)[1])
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"4x5:RGB", decoded)

    def test_image_messages_and_body(self):
        backend = self._backend()
        messages = backend._image_messages("data:image/png;base64,abc")
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[0]["content"][0]["text"], "Represent the user's input.")
        self.assertEqual(messages[1]["content"][0]["image_url"]["url"], "data:image/png;base64,abc")
        self.assertEqual(messages[1]["content"][1]["text"], "")
        self.assertEqual(messages[2]["content"][0]["text"], "")
        self.assertEqual(backend._embedding_body(messages), {
            "model": "fake-qwen",
            "messages": messages,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        })

    def test_batch_encoding_keeps_order_and_truncates(self):
        backend = self._backend(output_dim=2)
        calls = []
        embeddings = [[1, 2, 99], [3, 4, 99], [5, 6, 99]]
        def fake_post(messages, *, index=None):
            calls.append(index)
            return backend._validate_embedding_response(types.SimpleNamespace(data=[types.SimpleNamespace(embedding=embeddings[index])]), index=index)
        backend._post_embedding = fake_post
        features = backend._encode_messages_batch([[{"i": i}] for i in range(3)])
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(features.shape, (3, 2))
        self.assertEqual(features.device, "cpu")
        self.assertEqual(features.data, [[1, 2], [3, 4], [5, 6]])
        internal, image_features = backend.encode_image_with_internal([create_index.Image.Image(), create_index.Image.Image(), create_index.Image.Image()])
        self.assertEqual(internal.shape, image_features.shape)

    def test_response_validation_errors(self):
        backend = self._backend(output_dim=3)
        with self.assertRaisesRegex(RuntimeError, "any data"):
            backend._validate_embedding_response(types.SimpleNamespace(data=[]), index=0)
        with self.assertRaisesRegex(RuntimeError, "empty"):
            backend._validate_embedding_response(types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[])]), index=0)
        with self.assertRaisesRegex(RuntimeError, "smaller"):
            backend._validate_embedding_response(types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[1, 2])]), index=0)
        with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
            backend._validate_embedding_response(types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[1, float("nan"), 3])]), index=0)
        with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
            backend._validate_embedding_response(types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[1, float("inf"), 3])]), index=0)
        with self.assertRaises(TypeError):
            backend.encode_image_with_internal(["not-image"])

    def test_text_messages(self):
        backend = self._backend()
        messages = backend._text_messages("hello")
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "hello"}])

    def test_pil_image_inputs_bypass_default_collate(self):
        tensor = create_index.torch.FakeTensor
        images = [create_index.Image.Image(), create_index.Image.Image()]
        samples = [(images[0], None, 0, tensor([1]), tensor([2])), (images[1], None, 1, tensor([3]), tensor([4]))]
        search_batch, metadata, indices, wd, z3d = safe_collate(samples)
        self.assertEqual(search_batch, images)
        self.assertIsNone(metadata)
        self.assertEqual(indices, [0, 1])
        self.assertEqual(wd.shape, (2, 1))
        self.assertEqual(z3d.shape, (2, 1))

    def test_openclip_tensor_inputs_stay_on_default_collate_path(self):
        tensor = create_index.torch.FakeTensor
        samples = [(tensor([[1, 2], [3, 4]]), None, 0, tensor([1]), tensor([2])), (tensor([[5, 6], [7, 8]]), None, 1, tensor([3]), tensor([4]))]
        search_batch, _, _, _, _ = safe_collate(samples)
        self.assertEqual(search_batch.shape, (2, 2, 2))

    def test_cli_defaults_and_backend_args(self):
        original_argv = sys.argv[:]
        try:
            sys.argv = ["create_index.py"]
            args = create_index.parse_arguments()
            self.assertIsNone(args.qwen_max_pixels)
            self.assertEqual(args.qwen_api_base, "http://127.0.0.1:8000/v1")
            self.assertEqual(args.qwen_api_key, "EMPTY")
            self.assertEqual(args.qwen_api_timeout, 300.0)
            self.assertEqual(args.qwen_api_max_retries, 2)
            self.assertEqual(args.qwen_api_concurrency, 4)
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
