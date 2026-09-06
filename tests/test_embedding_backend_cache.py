import pathlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from app.domain.domain_object import ModelId
from app.domain.errors import SearchInputError
from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.model_metadata import SearchModelMetadata, write_model_metadata


class EmbeddingBackendCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        self.accessor = LocalAccessor(self.root)

    def register(self, model_id, directory):
        path = self.root / directory
        path.mkdir()
        (path / "metafiles.index").write_bytes(b"registered")
        (path / "sqlite_image_meta.db").write_bytes(b"registered")
        write_model_metadata(path, SearchModelMetadata(model_id.model_name, model_id.pretrained, directory))

    def test_simultaneous_first_requests_construct_one_backend(self):
        model = ModelId("ViT-B-32", "openai")
        self.register(model, "model")
        barrier = threading.Barrier(4)
        registration_barrier = threading.Barrier(4)
        resolve_directory = self.accessor._search_model_meta_dir
        started = threading.Event()
        release = threading.Event()
        backend = object()

        def construct(*args):
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release model constructor")
            return backend

        def load(_):
            barrier.wait(timeout=5)
            return self.accessor.load_embedding_backend(model)

        def resolve(model_id):
            path = resolve_directory(model_id)
            # Every worker has already missed the cache before construction starts.
            registration_barrier.wait(timeout=5)
            return path

        with patch("app.infrastructure.local_accessor.OpenClipEmbeddingBackend", side_effect=construct) as constructor, patch.object(self.accessor, "_search_model_meta_dir", side_effect=resolve):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(load, i) for i in range(4)]
                try:
                    self.assertTrue(started.wait(5))
                finally:
                    release.set()
                results = [future.result(timeout=5) for future in futures]
        constructor.assert_called_once_with("ViT-B-32", "openai")
        self.assertTrue(all(result is backend for result in results))

    def test_failed_construction_can_be_retried(self):
        model = ModelId("ViT-B-32", "openai")
        self.register(model, "model")
        expected = object()
        with patch("app.infrastructure.local_accessor.OpenClipEmbeddingBackend", side_effect=[RuntimeError("temporary failure"), expected]) as constructor:
            with self.assertRaisesRegex(RuntimeError, "temporary failure"):
                self.accessor.load_embedding_backend(model)
            self.assertIs(self.accessor.load_embedding_backend(model), expected)
            self.assertIs(self.accessor.load_embedding_backend(model), expected)
        self.assertEqual(constructor.call_count, 2)

    def test_qwen_legacy_pretrained_alias_shares_the_backend(self):
        model = ModelId("Qwen/Qwen3-VL-Embedding-2B", "")
        self.register(model, "qwen")
        with patch("app.infrastructure.local_accessor.QwenEmbeddingBackend", return_value=object()) as constructor:
            first = self.accessor.load_embedding_backend(ModelId(model.model_name, "openai"))
            self.assertIs(self.accessor.load_embedding_backend(model), first)
        constructor.assert_called_once_with(model.model_name)

    def test_unregistered_requests_do_not_allocate_model_locks(self):
        with self.assertRaises(SearchInputError):
            self.accessor.load_embedding_backend(ModelId("unknown", ""))
        self.assertEqual(self.accessor._embedding_backend_locks, {})
        self.assertEqual(self.accessor._embedding_backends, {})


if __name__ == "__main__":
    unittest.main()
