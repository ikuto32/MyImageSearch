import errno
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from app.domain.domain_object import Image
from app.domain.errors import ResourceLimitError
from app.infrastructure import zip_archive
from app.infrastructure.zip_archive import create_zip


def image(name: str, binary: bytes = b"image contents") -> Image:
    return Image(binary, "image/png", name)


class ZipArchiveTests(unittest.TestCase):
    def test_archive_has_unique_safe_names_and_rolls_to_disk(self):
        names = ("a/photo.jpg", "b\\photo.jpg", "../photo.jpg", "PHOTO.JPG", "")
        with patch.object(zip_archive, "ZIP_MEMORY_BYTES", 1):
            with create_zip(image(name, b"bytes") for name in names) as result:
                self.assertTrue(result._rolled)
                with zipfile.ZipFile(result) as archive:
                    self.assertEqual(
                        archive.namelist(),
                        ["photo.jpg", "photo (2).jpg", "photo (3).jpg", "PHOTO (4).JPG", "image"],
                    )
                    self.assertEqual(archive.read("photo (3).jpg"), b"bytes")

    def test_archive_writes_all_1024_entries_with_bounded_spool(self):
        with create_zip(image(f"{i:04d}.png") for i in range(1024)) as stream:
            self.assertEqual(stream.tell(), 0)
            with zipfile.ZipFile(stream) as archive:
                self.assertEqual(len(archive.infolist()), 1024)
                self.assertEqual(archive.namelist()[-1], "1023.png")

    def test_archive_budget_closes_temporary_file(self):
        spool = tempfile.SpooledTemporaryFile()
        with patch.object(zip_archive.tempfile, "SpooledTemporaryFile", return_value=spool):
            with patch.object(zip_archive, "MAX_ZIP_BYTES", 4):
                with self.assertRaises(ResourceLimitError):
                    create_zip([image("x.png", b"12345")])
        self.assertTrue(spool.closed)

    def test_disk_capacity_failure_is_detected_before_member_write_and_closes_spool(self):
        spool = tempfile.SpooledTemporaryFile(max_size=1)
        with (
            patch.object(zip_archive, "ZIP_MEMORY_BYTES", 1),
            patch.object(zip_archive.tempfile, "SpooledTemporaryFile", return_value=spool),
            patch.object(zip_archive.shutil, "disk_usage", return_value=SimpleNamespace(free=0)),
            patch.object(zip_archive.zipfile.ZipFile, "writestr") as write,
        ):
            with self.assertRaises(ResourceLimitError):
                create_zip([image("a.png")])
        write.assert_not_called()
        self.assertTrue(spool.closed)

    def test_disk_full_race_becomes_resource_error_and_closes_spool(self):
        spool = tempfile.SpooledTemporaryFile()
        with (
            patch.object(zip_archive.tempfile, "SpooledTemporaryFile", return_value=spool),
            patch.object(zip_archive.zipfile.ZipFile, "writestr", side_effect=OSError(errno.ENOSPC, "disk full")),
        ):
            with self.assertRaises(ResourceLimitError):
                create_zip([image("a.png")])
        self.assertTrue(spool.closed)

    def test_failure_while_reading_images_closes_spool(self):
        spool = tempfile.SpooledTemporaryFile()

        def images():
            yield image("a.png")
            raise RuntimeError("source failed")

        with patch.object(zip_archive.tempfile, "SpooledTemporaryFile", return_value=spool):
            with self.assertRaisesRegex(RuntimeError, "source failed"):
                create_zip(images())
        self.assertTrue(spool.closed)


if __name__ == "__main__":
    unittest.main()
