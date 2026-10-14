from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from app.domain.domain_object import ModelId
from app.domain.errors import SearchInputError
from app.infrastructure import local_accessor
from app.infrastructure.index_files import IndexFiles
from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.model_registry import ModelRegistry
from app.infrastructure.once_cache import OnceCache
from helpers import create_catalog


class OnceCacheTests(unittest.TestCase):
    def test_simultaneous_first_requests_build_one_value(self):
        cache = OnceCache()
        barrier = threading.Barrier(4)
        started = threading.Event()
        release = threading.Event()
        builds = []

        def build():
            builds.append(threading.get_ident())
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release the build")
            return object()

        def load(_):
            barrier.wait(timeout=5)
            return cache.get("key", build)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load, i) for i in range(4)]
            try:
                self.assertTrue(started.wait(5))
            finally:
                release.set()
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual(len(builds), 1)
        self.assertTrue(all(result is results[0] for result in results))

    def test_failed_build_is_not_cached_and_can_be_retried(self):
        cache = OnceCache()
        expected = object()
        outcomes = iter([RuntimeError("temporary failure"), expected])

        def build():
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with self.assertRaisesRegex(RuntimeError, "temporary failure"):
            cache.get("key", build)
        self.assertNotIn("key", cache)
        self.assertIs(cache.get("key", build), expected)
        self.assertIs(cache.get("key", build), expected)

    def test_different_keys_do_not_wait_for_each_other(self):
        cache = OnceCache()
        release = threading.Event()
        started = threading.Event()

        def slow():
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release the slow build")
            return "slow"

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cache.get, "slow", slow)
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(cache.get("fast", lambda: "fast"), "fast")
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5), "slow")


class LocalAccessorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model = ModelId("ViT-B-32", "openai")
        files = create_catalog(Path(temporary.name), "ViT-B-32-openai", [])
        self.files = IndexFiles(files.directory.resolve())
        self.accessor = LocalAccessor(ModelRegistry(Path(temporary.name)))

    def test_each_model_loads_its_backend_and_index_once(self):
        backend, index = object(), object()
        with (
            patch.object(local_accessor, "create_embedding_backend", return_value=backend) as create,
            patch.object(local_accessor, "load_search_index", return_value=index) as load,
        ):
            for _ in range(2):
                self.assertIs(self.accessor.load_embedding_backend(self.model), backend)
                self.assertIs(self.accessor.load_search_index(self.model), index)
        create.assert_called_once_with(self.model)
        load.assert_called_once_with(self.files)

    def test_failed_loads_can_be_retried(self):
        expected = object()
        with patch.object(local_accessor, "load_search_index", side_effect=[OSError("unavailable"), expected]):
            with self.assertRaises(OSError):
                self.accessor.load_search_index(self.model)
            self.assertIs(self.accessor.load_search_index(self.model), expected)

    def test_unregistered_models_are_rejected_before_loading_or_caching_anything(self):
        with (
            patch.object(local_accessor, "create_embedding_backend") as create,
            patch.object(local_accessor, "load_search_index") as load,
        ):
            for model in (ModelId("unknown/Qwen-model", ""), ModelId("ViT-B-32", "other")):
                with self.assertRaisesRegex(SearchInputError, "not registered"):
                    self.accessor.load_embedding_backend(model)
                with self.assertRaisesRegex(SearchInputError, "not registered"):
                    self.accessor.load_search_index(model)
        create.assert_not_called()
        load.assert_not_called()
        self.assertEqual(self.accessor._embedding_backends._locks, {})
        self.assertEqual(self.accessor._search_indexes._locks, {})


if __name__ == "__main__":
    unittest.main()
