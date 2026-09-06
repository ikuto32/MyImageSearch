import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image as PILImage

from app.domain.domain_object import Image, ImageId, ImageName
from app.domain.errors import ResourceLimitError
from app.infrastructure.local_repository import LocalRepository


class RepositoryFileLimitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'images'
        self.root.mkdir()
        self.repo = LocalRepository(self.root, Path(self.temp.name) / 'meta')
        self.item_id = ImageId('image')

    def set_path(self, value):
        self.repo.set_image_paths({self.item_id: Path(value)})

    def test_root_escape_absolute_and_unsupported_files_are_rejected(self):
        outside = Path(self.temp.name) / 'outside.jpg'
        outside.write_bytes(b'private')
        (self.root / 'private.txt').write_text('private')
        for path in ('../outside.jpg', outside, 'private.txt'):
            self.set_path(path)
            for load in (self.repo.load_image, self.repo.load_small_image):
                with self.subTest(path=str(path), load=load.__name__), self.assertRaises(ValueError):
                    load(self.item_id)

    def test_missing_and_broken_images_fail_cleanly(self):
        self.set_path('missing.png')
        for load in (self.repo.load_image, self.repo.load_small_image):
            with self.assertRaises(ValueError):
                load(self.item_id)
        self.set_path('broken.png')
        (self.root / 'broken.png').write_bytes(b'invalid image')
        with self.assertRaises(ValueError):
            self.repo.load_small_image(self.item_id)

    def test_file_and_pixel_budgets_apply_before_decoding(self):
        PILImage.new('RGB', (12, 10)).save(self.root / 'image.png')
        self.set_path('image.png')
        with patch.object(self.repo, 'MAX_IMAGE_BYTES', 1):
            for load in (self.repo.load_image, self.repo.load_small_image):
                with self.assertRaises(ResourceLimitError):
                    load(self.item_id)
        with patch.object(self.repo, 'MAX_IMAGE_PIXELS', 119):
            with self.assertRaises(ResourceLimitError):
                self.repo.load_small_image(self.item_id)

    def test_thumbnail_applies_exif_rotation(self):
        picture = PILImage.new('RGB', (40, 20))
        exif = picture.getexif()
        exif[274] = 6
        picture.save(self.root / 'rotated.jpg', exif=exif)
        self.set_path('rotated.jpg')
        result = self.repo.load_small_image(self.item_id)
        with PILImage.open(io.BytesIO(result.binary)) as thumbnail:
            self.assertEqual(thumbnail.size, (20, 40))

    def test_inference_image_applies_pixel_and_byte_budgets(self):
        data = io.BytesIO()
        PILImage.new('RGB', (12, 10)).save(data, format='PNG')
        image = Image(data.getvalue(), 'image/png')
        with patch('app.domain.domain_object.MAX_IMAGE_BYTES', 1):
            with self.assertRaises(ResourceLimitError):
                image.to_ptl_image()
        with patch('app.domain.domain_object.MAX_IMAGE_PIXELS', 119):
            with self.assertRaises(ResourceLimitError):
                image.to_ptl_image()

    def test_inference_image_preserves_index_orientation_and_owns_pixels(self):
        picture = PILImage.new('RGB', (40, 20))
        exif = picture.getexif()
        exif[274] = 6
        data = io.BytesIO()
        picture.save(data, format='JPEG', exif=exif)
        with Image(data.getvalue(), 'image/jpeg').to_ptl_image() as decoded:
            self.assertEqual(decoded.size, (40, 20))
            self.assertEqual(decoded.getpixel((0, 0)), (0, 0, 0))

    def test_archive_has_unique_safe_names_and_rolls_to_disk(self):
        def items():
            for name in ('a/photo.jpg', 'b/photo.jpg', '../photo.jpg'):
                yield Image(b'bytes', 'image/jpeg'), ImageName(name)
        with patch.object(self.repo, 'ZIP_MEMORY_BYTES', 1):
            with self.repo.create_zip_from_images(items()) as result:
                self.assertTrue(result._rolled)
                with zipfile.ZipFile(result) as archive:
                    self.assertEqual(archive.namelist(), ['photo.jpg', 'photo (2).jpg', 'photo (3).jpg'])
                    self.assertEqual(archive.read('photo (3).jpg'), b'bytes')

    def test_archive_budget_closes_temporary_file(self):
        spool = tempfile.SpooledTemporaryFile()
        items = [(Image(b'12345', 'image/png'), ImageName('x.png'))]
        with patch('app.infrastructure.local_repository.tempfile.SpooledTemporaryFile', return_value=spool):
            with patch.object(self.repo, 'MAX_ZIP_BYTES', 4):
                with self.assertRaises(ResourceLimitError):
                    self.repo.create_zip_from_images(items)
        self.assertTrue(spool.closed)
