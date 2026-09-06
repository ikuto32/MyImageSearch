import io
import pathlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from PIL import Image as PILImage

from app.domain.domain_object import ImageId
from app.infrastructure.local_repository import LocalRepository


class LocalRepositoryCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = pathlib.Path(self.temp_dir.name)
        self.ids = [ImageId(str(index)) for index in range(3)]
        self.paths = {}
        for image_id, color in zip(self.ids, ["red", "green", "blue"]):
            path = pathlib.Path(f"{image_id.id}.bmp")
            PILImage.new("RGB", (20, 20), color).save(self.root / path)
            self.paths[image_id] = path
        self.image_size = (self.root / self.paths[self.ids[0]]).stat().st_size

    def repository(self, budget):
        repo = LocalRepository(self.root, thumbnail_cache_bytes=budget)
        repo.set_image_paths(self.paths)
        return repo

    def test_evicts_least_recently_used_thumbnail_within_byte_budget(self):
        repo = self.repository(self.image_size * 2)
        first = repo.load_small_image(self.ids[0])
        second = repo.load_small_image(self.ids[1])
        self.assertIs(repo.load_small_image(self.ids[0]), first)

        repo.load_small_image(self.ids[2])

        self.assertIs(repo.load_small_image(self.ids[0]), first)
        self.assertIsNot(repo.load_small_image(self.ids[1]), second)
        self.assertLessEqual(repo._thumbnail_cache_size, self.image_size * 2)
        self.assertEqual(repo._thumbnail_cache_size, sum(
            len(image.binary) for image in repo._thumbnail_cache.values()
        ))

    def test_oversized_and_disabled_caches_still_return_images(self):
        for budget in [0, self.image_size - 1]:
            with self.subTest(budget=budget):
                repo = self.repository(budget)
                first = repo.load_small_image(self.ids[0])
                second = repo.load_small_image(self.ids[0])
                self.assertEqual(first.binary, second.binary)
                self.assertIsNot(first, second)
                self.assertEqual(repo._thumbnail_cache_size, 0)
                self.assertFalse(repo._thumbnail_cache)

    def test_thumbnail_keeps_aspect_ratio_and_mime(self):
        PILImage.new("RGB", (1000, 500), "red").save(self.root / self.paths[self.ids[0]])
        result = self.repository(1024 * 1024).load_small_image(self.ids[0])
        with PILImage.open(io.BytesIO(result.binary)) as image:
            self.assertEqual(image.size, (400, 200))
        self.assertEqual(result.content_type, "image/bmp")

    def test_original_image_is_read_again_instead_of_retained(self):
        repo = self.repository(self.image_size)
        first = repo.load_image(self.ids[0])
        path = self.root / self.paths[self.ids[0]]
        PILImage.new("RGB", (20, 20), "black").save(path)
        second = repo.load_image(self.ids[0])
        self.assertNotEqual(first.binary, second.binary)
        self.assertEqual(second.binary, path.read_bytes())

    def test_replacing_paths_invalidates_thumbnails_and_names(self):
        repo = self.repository(self.image_size * 2)
        first = repo.load_small_image(self.ids[0])
        repo.get_image_name(self.ids[0])
        repo.set_image_paths({self.ids[0]: self.paths[self.ids[1]]})
        self.assertEqual(repo._thumbnail_cache_size, 0)
        self.assertNotEqual(repo.load_small_image(self.ids[0]).binary, first.binary)
        self.assertEqual(repo.get_image_name(self.ids[0]).name, "1.bmp")
        with self.assertRaises(ValueError):
            repo.load_small_image(self.ids[1])

    def test_image_generated_during_path_replacement_does_not_repopulate_cache(self):
        repo = self.repository(self.image_size * 2)
        started = threading.Event()
        release = threading.Event()
        original_open = PILImage.open

        def blocked_open(path):
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release thumbnail generation")
            return original_open(path)

        with ThreadPoolExecutor(max_workers=1) as executor:
            with patch("app.infrastructure.local_repository.PILImage.open", side_effect=blocked_open):
                future = executor.submit(repo.load_small_image, self.ids[0])
                try:
                    self.assertTrue(started.wait(5))
                    repo.set_image_paths({self.ids[0]: self.paths[self.ids[1]]})
                finally:
                    release.set()
                old_image = future.result(timeout=5)

        self.assertFalse(repo._thumbnail_cache)
        self.assertNotEqual(repo.load_small_image(self.ids[0]).binary, old_image.binary)

    def test_parallel_requests_account_for_each_cached_image_once(self):
        repo = self.repository(self.image_size * 2)
        original_open = PILImage.open
        barrier = threading.Barrier(4)

        def concurrent_open(path):
            barrier.wait(timeout=5)
            return original_open(path)

        with patch("app.infrastructure.local_repository.PILImage.open", side_effect=concurrent_open):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(repo.load_small_image, [self.ids[0]] * 4))

        self.assertEqual(repo._thumbnail_cache_size, len(results[0].binary))
        self.assertEqual(len(repo._thumbnail_cache), 1)

    def test_negative_byte_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            self.repository(-1)


if __name__ == "__main__":
    unittest.main()
