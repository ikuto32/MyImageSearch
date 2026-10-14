import io
import unittest
from unittest.mock import Mock

from app.application.usecase import Usecase
from app.domain.domain_object import Image, ImageId, ImageItem, ModelId
from app.domain.errors import ImageUnavailableError, ResourceLimitError, SearchInputError


class UsecaseCatalogTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelId("model", "weights")
        self.repository = Mock()
        self.accessor = Mock()
        self.usecase = Usecase(self.repository, self.accessor)

    def tearDown(self):
        # Browsing and downloading never need FAISS or an embedding model.
        self.assertEqual(self.accessor.mock_calls, [])

    def test_catalog_requests_pass_a_normalized_rating_filter_to_the_repository(self):
        self.usecase.get_image_items_by_page(self.model, 2, 60, ["general", "explicit", "general"])
        self.repository.load_image_items_by_page.assert_called_once_with(self.model, 2, 60, ("explicit", "general"))
        self.usecase.get_catalog_count(self.model)
        self.repository.count_image_items.assert_called_once_with(self.model, None)
        self.usecase.get_catalog_count(self.model, [])
        self.repository.count_image_items.assert_called_with(self.model, ())

    def test_malformed_ratings_fail_before_any_catalog_scan(self):
        for ratings in ("general", [None], ["unknown"], [{}]):
            with self.subTest(ratings=ratings):
                with self.assertRaises(SearchInputError):
                    self.usecase.get_catalog_count(self.model, ratings)
                with self.assertRaises(SearchInputError):
                    self.usecase.get_image_items_by_page(self.model, 0, 60, ratings)
                with self.assertRaises(SearchInputError):
                    self.usecase.get_download_ids(self.model, 1, ratings)
        self.assertEqual(self.repository.mock_calls, [])

    def test_download_ids_are_the_first_rows_of_the_filtered_catalog(self):
        self.repository.load_image_items_by_page.return_value = [
            ImageItem(ImageId("a"), "a.png"), ImageItem(ImageId("b"), "b.png"),
        ]
        self.assertEqual(self.usecase.get_download_ids(self.model, 1024, ["general"]), ["a", "b"])
        self.repository.load_image_items_by_page.assert_called_once_with(self.model, 0, 1024, ("general",))

    def test_download_limit_is_validated_before_querying(self):
        for limit in (0, -1, 1025, True, "10"):
            with self.subTest(limit=limit), self.assertRaises(SearchInputError):
                self.usecase.get_download_ids(self.model, limit)
        self.assertEqual(self.repository.mock_calls, [])


class UsecaseArchiveTests(unittest.TestCase):
    def setUp(self):
        self.repository = Mock()
        self.repository.load_image.side_effect = self.load_image
        self.repository.create_zip_from_images.side_effect = self.create_zip
        self.archived = []
        self.usecase = Usecase(self.repository, Mock())

    @staticmethod
    def load_image(image_id):
        if image_id.startswith("missing"):
            raise ImageUnavailableError(f"unknown image: {image_id}")
        if image_id == "huge":
            raise ResourceLimitError("too large")
        return Image(f"binary:{image_id}".encode(), "image/png", f"{image_id}.png")

    def create_zip(self, images):
        self.archived = [(image.binary, image.name) for image in images]
        self.stream = io.BytesIO(b"zip")
        return self.stream

    def test_unavailable_images_are_skipped_and_counted(self):
        with self.assertLogs("app.application.usecase", level="WARNING") as logs:
            archive = self.usecase.get_images_zip(["valid_1", "missing_1", "valid_2", "missing_2"])
        self.assertEqual(len(logs.records), 2)
        self.assertIs(archive.stream, self.stream)
        self.assertEqual(self.archived, [(b"binary:valid_1", "valid_1.png"), (b"binary:valid_2", "valid_2.png")])
        self.assertEqual((archive.requested_count, archive.image_count, archive.skipped_count), (4, 2, 2))

    def test_images_are_loaded_on_demand_while_the_archive_is_written(self):
        def consume(images):
            self.repository.load_image.assert_not_called()
            next(images)
            self.assertEqual(self.repository.load_image.call_count, 1)
            next(images)
            self.assertEqual(self.repository.load_image.call_count, 2)
            return io.BytesIO()

        self.repository.create_zip_from_images.side_effect = consume
        self.assertEqual(self.usecase.get_images_zip(["0", "1"]).image_count, 2)

    def test_archive_without_any_image_fails_and_closes_its_stream(self):
        with self.assertLogs("app.application.usecase", level="WARNING"):
            with self.assertRaisesRegex(SearchInputError, "保存できる画像"):
                self.usecase.get_images_zip(["missing_1", "missing_2"])
        self.assertTrue(self.stream.closed)

    def test_resource_limits_abort_the_whole_archive(self):
        with self.assertRaises(ResourceLimitError):
            self.usecase.get_images_zip(["valid", "huge"])


if __name__ == "__main__":
    unittest.main()
