from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image as PILImage

from app.domain import limits
from app.domain.errors import ImageUnavailableError, ResourceLimitError
from app.infrastructure.image_store import ImageStore
from helpers import save_image


class ImageStoreTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "images"
        self.root.mkdir()


class ThumbnailCacheTests(ImageStoreTestCase):
    def setUp(self):
        super().setUp()
        self.names = ["0.bmp", "1.bmp", "2.bmp"]
        for name, color in zip(self.names, ["red", "green", "blue"]):
            save_image(self.root / name, color=color)
        self.image_size = (self.root / self.names[0]).stat().st_size

    def store(self, budget):
        return ImageStore(self.root, thumbnail_cache_bytes=budget)

    def assert_consistent_accounting(self, store):
        self.assertEqual(store._thumbnail_bytes, sum(len(image.binary) for image in store._thumbnails.values()))

    def test_evicts_least_recently_used_thumbnail_within_byte_budget(self):
        store = self.store(self.image_size * 2)
        first = store.load_thumbnail(self.names[0])
        second = store.load_thumbnail(self.names[1])
        self.assertIs(store.load_thumbnail(self.names[0]), first)

        store.load_thumbnail(self.names[2])

        self.assertIs(store.load_thumbnail(self.names[0]), first)
        self.assertIsNot(store.load_thumbnail(self.names[1]), second)
        self.assertLessEqual(store._thumbnail_bytes, self.image_size * 2)
        self.assert_consistent_accounting(store)

    def test_oversized_and_disabled_caches_still_return_images(self):
        for budget in [0, self.image_size - 1]:
            with self.subTest(budget=budget):
                store = self.store(budget)
                first = store.load_thumbnail(self.names[0])
                second = store.load_thumbnail(self.names[0])
                self.assertEqual(first.binary, second.binary)
                self.assertIsNot(first, second)
                self.assertEqual(store._thumbnail_bytes, 0)
                self.assertFalse(store._thumbnails)

    def test_parallel_requests_account_for_each_cached_image_once(self):
        store = self.store(self.image_size * 2)
        original_open = PILImage.open
        barrier = threading.Barrier(4)

        def concurrent_open(path):
            barrier.wait(timeout=5)
            return original_open(path)

        with patch("app.infrastructure.image_store.PILImage.open", side_effect=concurrent_open):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(store.load_thumbnail, [self.names[0]] * 4))

        self.assertEqual(store._thumbnail_bytes, len(results[0].binary))
        self.assertEqual(len(store._thumbnails), 1)

    def test_negative_byte_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store(-1)


class ImageFileTests(ImageStoreTestCase):
    def setUp(self):
        super().setUp()
        self.store = ImageStore(self.root)

    def test_thumbnail_keeps_aspect_ratio_format_and_name(self):
        save_image(self.root / "wide.bmp", size=(1000, 500))
        result = self.store.load_thumbnail("wide.bmp")
        with PILImage.open(io.BytesIO(result.binary)) as image:
            self.assertEqual(image.size, (400, 200))
            self.assertEqual(image.format, "BMP")
        self.assertEqual((result.content_type, result.name), ("image/bmp", "wide.bmp"))

    def test_thumbnail_of_a_format_that_cannot_be_written_is_a_png(self):
        (self.root / "icon.xpm").write_text(
            '/* XPM */\nstatic char * icon[] = {\n"2 2 2 1",\n"a c #ff0000",\n"b c #0000ff",\n"ab",\n"ba"};\n'
        )
        result = self.store.load_thumbnail("icon.xpm")
        self.assertEqual((result.content_type, result.name), ("image/png", "icon.xpm"))
        with PILImage.open(io.BytesIO(result.binary)) as image:
            self.assertEqual((image.format, image.size), ("PNG", (2, 2)))
            self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))

    def test_thumbnail_in_a_mode_its_format_cannot_write_is_a_png(self):
        # Pillow reads this texture format as RGB but only writes palette images to it.
        PILImage.new("P", (4, 2)).save(self.root / "texture.blp")
        result = self.store.load_thumbnail("texture.blp")
        self.assertEqual(result.content_type, "image/png")
        with PILImage.open(io.BytesIO(result.binary)) as image:
            self.assertEqual((image.format, image.size), ("PNG", (4, 2)))

    def test_thumbnail_of_a_cmyk_image_without_a_writer_is_an_rgb_png_without_the_cmyk_profile(self):
        width, height = 20, 10
        profile = b"cmyk"
        resources = b"8BIM" + struct.pack(">HHI", 0x040F, 0, len(profile)) + profile
        (self.root / "print.psd").write_bytes(
            b"8BPS" + struct.pack(">H6xHIIHH", 1, 4, height, width, 8, 4)  # 4 channels, 8 bits, CMYK
            + struct.pack(">I", 0) + struct.pack(">I", len(resources)) + resources
            + struct.pack(">IH", 0, 0) + bytes(4 * width * height)  # no layers, raw channel data
        )
        with PILImage.open(self.root / "print.psd") as source:
            self.assertEqual((source.format, source.mode, source.info.get("icc_profile")), ("PSD", "CMYK", profile))
        result = self.store.load_thumbnail("print.psd")
        self.assertEqual(result.content_type, "image/png")
        with PILImage.open(io.BytesIO(result.binary)) as image:
            self.assertEqual((image.format, image.mode, image.size), ("PNG", "RGB", (width, height)))
            self.assertNotIn("icc_profile", image.info)

    def test_thumbnail_applies_exif_rotation(self):
        picture = PILImage.new("RGB", (40, 20))
        exif = picture.getexif()
        exif[274] = 6
        picture.save(self.root / "rotated.jpg", exif=exif)
        result = self.store.load_thumbnail("rotated.jpg")
        with PILImage.open(io.BytesIO(result.binary)) as thumbnail:
            self.assertEqual(thumbnail.size, (20, 40))

    def test_original_image_is_read_again_instead_of_retained(self):
        path = save_image(self.root / "sub" / "image.png", color="red")
        first = self.store.load_original("sub/image.png")
        save_image(path, color="black")
        second = self.store.load_original("sub/image.png")
        self.assertNotEqual(first.binary, second.binary)
        self.assertEqual(second.binary, path.read_bytes())
        self.assertEqual((second.content_type, second.name), ("image/png", "sub/image.png"))

    def test_root_escape_absolute_and_unsupported_files_are_rejected(self):
        outside = Path(self.root.parent) / "outside.jpg"
        outside.write_bytes(b"private")
        (self.root / "private.txt").write_text("private")
        for path in ("../outside.jpg", str(outside), "private.txt", "C:\\outside.jpg", "\\\\server\\share\\a.jpg"):
            for load in (self.store.load_original, self.store.load_thumbnail):
                with self.subTest(path=path, load=load.__name__), self.assertRaises(ImageUnavailableError):
                    load(path)

    def test_missing_and_broken_images_fail_cleanly(self):
        for load in (self.store.load_original, self.store.load_thumbnail):
            with self.assertRaises(ImageUnavailableError):
                load("missing.png")
        (self.root / "broken.png").write_bytes(b"invalid image")
        with self.assertRaises(ImageUnavailableError):
            self.store.load_thumbnail("broken.png")

    def test_file_and_pixel_budgets_apply_before_decoding(self):
        save_image(self.root / "image.png", size=(12, 10))
        with patch.object(limits, "MAX_IMAGE_BYTES", 1):
            for load in (self.store.load_original, self.store.load_thumbnail):
                with self.assertRaises(ResourceLimitError):
                    load("image.png")
        with patch.object(limits, "MAX_IMAGE_PIXELS", 119):
            with self.assertRaises(ResourceLimitError):
                self.store.load_thumbnail("image.png")
            self.assertEqual(self.store.load_original("image.png").name, "image.png")


if __name__ == "__main__":
    unittest.main()
