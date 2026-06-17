import sqlite3
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen_collate import _install_dummy_modules  # noqa: E402

_install_dummy_modules()

import create_index  # noqa: E402

torch = create_index.torch


def _patch_fake_tensor():
    FakeTensor = torch.FakeTensor

    def _rows(shape, fill=0.0):
        if len(shape) == 1:
            return [fill for _ in range(shape[0])]
        return [_rows(shape[1:], fill) for _ in range(shape[0])]

    def zeros(*shape, dtype=None, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return FakeTensor(_rows(shape, 0.0))

    def ones(*shape, dtype=None, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return FakeTensor(_rows(shape, 1.0))

    def arange(n):
        return FakeTensor(list(range(n)))

    def norm(self, dim=None, keepdim=False):
        if dim in (-1, 1) and self.ndim == 2:
            data = [[1.0] for _ in range(self.shape[0])] if keepdim else [1.0 for _ in range(self.shape[0])]
            return FakeTensor(data)
        return FakeTensor([1.0])

    def tobytes(self):
        return b"x" * (4 * self.shape[0])

    def min_method(self, dim=None):
        if dim == 1:
            return FakeTensor([0.0 for _ in range(self.shape[0])]), FakeTensor([0 for _ in range(self.shape[0])])
        return FakeTensor([0.0]), FakeTensor([0])

    FakeTensor.norm = norm
    FakeTensor.clamp_min = lambda self, value: self
    FakeTensor.__truediv__ = lambda self, other: self
    FakeTensor.contiguous = lambda self: self
    FakeTensor.cpu = lambda self: self
    FakeTensor.float = lambda self: self
    FakeTensor.detach = lambda self: self
    FakeTensor.numpy = lambda self: self
    FakeTensor.tolist = lambda self: self.data
    FakeTensor.__iter__ = lambda self: (FakeTensor(item) for item in self.data)
    FakeTensor.tobytes = tobytes
    FakeTensor.__len__ = lambda self: self.shape[0]
    FakeTensor.__int__ = lambda self: int(self.data)
    FakeTensor.__float__ = lambda self: float(self.data)
    FakeTensor.min = min_method
    torch.zeros = zeros
    torch.ones = ones
    torch.arange = arange
    torch.from_numpy = lambda value: FakeTensor(value)
    torch.cdist = lambda features, centers: zeros(features.shape[0], centers.shape[0])

    class _NoGrad:
        def __call__(self, fn=None):
            return self if fn is None else fn
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False

    torch.no_grad = _NoGrad()
    torch.inference_mode = _NoGrad()


_patch_fake_tensor()


class _SearchModel:
    def encode_image_with_internal(self, image_input):
        features = torch.ones(image_input.shape[0], 4)
        return features, features


class _MetadataModel:
    def encode_image_with_internal(self, metadata_input):
        return torch.ones(metadata_input.shape[0], 1024), torch.ones(metadata_input.shape[0], 768)


class _AestheticModel:
    def __init__(self):
        self.param = types.SimpleNamespace(dtype=torch.float32)
        self.seen_shape = None

    def parameters(self):
        return iter([self.param])

    def score_batch(self, features):
        self.seen_shape = tuple(features.shape)
        return torch.zeros(features.shape[0], 1)


class _PonyScorer:
    def __init__(self):
        self.seen_shape = None

    def score_batch(self, features):
        self.seen_shape = tuple(features.shape)
        return [0.25] * features.shape[0]


class _StyleCluster:
    def __init__(self):
        self.cluster_centers = torch.zeros(2048, 1024)
        self.seen_shape = None

    def get_cluster_batch(self, features):
        self.seen_shape = tuple(features.shape)
        return [7] * features.shape[0], [0.0] * features.shape[0]


class _TaggingService:
    def tag_batch(self, wd_batch, z3d_batch):
        return [{"rating": "general", "tags": ["tag"]} for _ in range(len(wd_batch))]


class FeatureRoutingTests(unittest.TestCase):
    def test_extract_image_features_routes_metadata_feature_spaces_separately(self):
        con = sqlite3.connect(":memory:")
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE image_meta (
                image_id TEXT PRIMARY KEY, image_path TEXT, meta BLOB,
                aesthetic_quality REAL, pony_aesthetic_quality REAL,
                style_cluster TEXT, rating TEXT, image_tags TEXT,
                time_stamp_ISO TEXT, search_embedding_model TEXT,
                metadata_embedding_model TEXT)
        """)
        args = types.SimpleNamespace(
            batch_size=3, search_model_out_dim=None, use_existing_tags=False,
            image_dir=".", search_backend="qwen3_vl", search_model_id="fake-qwen",
            disable_clip_metadata=False,
        )
        batch_size = 3
        loader = [(torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), torch.arange(batch_size), torch.zeros(batch_size, 1), torch.zeros(batch_size, 1))]
        aesthetic_model = _AestheticModel()
        pony_scorer = _PonyScorer()
        style_cluster = _StyleCluster()
        old_tqdm = getattr(create_index.tqdm, "tqdm", None)
        create_index.tqdm.tqdm = lambda iterable, total=None: iterable
        try:
            create_index.extract_image_features(
                args, "cpu", _SearchModel(), con, cur, loader,
                [f"image-{i}.png" for i in range(batch_size)],
                [f"id-{i}" for i in range(batch_size)],
                aesthetic_model, pony_scorer, style_cluster, _TaggingService(),
                metadata_model=_MetadataModel(),
            )
        finally:
            if old_tqdm is None:
                delattr(create_index.tqdm, "tqdm")
            else:
                create_index.tqdm.tqdm = old_tqdm

        self.assertEqual(aesthetic_model.seen_shape, (batch_size, 768))
        self.assertEqual(pony_scorer.seen_shape, (batch_size, 768))
        self.assertEqual(style_cluster.seen_shape, (batch_size, 1024))
        self.assertEqual(cur.execute("SELECT COUNT(*) FROM image_meta").fetchone()[0], batch_size)

    def test_style_cluster_dimension_mismatch_has_clear_error(self):
        cluster = create_index.StyleCluster.__new__(create_index.StyleCluster)
        cluster.cluster_centers = torch.zeros(2048, 1024)
        cluster.kept_cluster_indices = list(range(2048))
        with self.assertRaisesRegex(ValueError, r"features=768.*cluster_centers=1024"):
            cluster.get_cluster_batch(torch.zeros(2, 768))

    def test_style_cluster_batch_converts_dtype_before_cdist_on_cpu(self):
        cluster = create_index.StyleCluster.__new__(create_index.StyleCluster)
        cluster.cluster_centers = torch.zeros(2, 4)
        cluster.kept_cluster_indices = [10, 11]
        cluster_ids, distances = cluster.get_cluster_batch(torch.ones(3, 4))
        self.assertEqual(cluster_ids, [10, 10, 10])
        self.assertEqual(distances.data, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
