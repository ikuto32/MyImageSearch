import io
import json
import unittest
from unittest.mock import patch

from PIL import Image as PILImage

from app.domain import limits
from app.domain.domain_object import Image, ModelId
from app.domain.errors import ResourceLimitError
from app.domain.rating import rating_category
from app.domain.search_query import InvalidSearchQuery, SearchQuery, parse_search_query


def encoded(picture: PILImage.Image, image_format: str, **options) -> bytes:
    data = io.BytesIO()
    picture.save(data, format=image_format, **options)
    return data.getvalue()


class ImageTests(unittest.TestCase):
    def test_inference_image_applies_pixel_and_byte_budgets(self):
        image = Image(encoded(PILImage.new("RGB", (12, 10)), "PNG"), "image/png")
        with patch.object(limits, "MAX_IMAGE_BYTES", 1):
            with self.assertRaises(ResourceLimitError):
                image.to_pil_image()
        with patch.object(limits, "MAX_IMAGE_PIXELS", 119):
            with self.assertRaises(ResourceLimitError):
                image.to_pil_image()
        with image.to_pil_image() as decoded:
            self.assertEqual(decoded.size, (12, 10))

    def test_inference_image_preserves_index_orientation_and_owns_pixels(self):
        picture = PILImage.new("RGB", (40, 20))
        exif = picture.getexif()
        exif[274] = 6
        with Image(encoded(picture, "JPEG", exif=exif), "image/jpeg").to_pil_image() as decoded:
            self.assertEqual(decoded.size, (40, 20))
            self.assertEqual(decoded.getpixel((0, 0)), (0, 0, 0))


class RatingTests(unittest.TestCase):
    def test_unknown_and_unset_ratings_are_unclassified(self):
        self.assertEqual([rating_category(value) for value in ("general", "explicit")], ["general", "explicit"])
        self.assertEqual({rating_category(value) for value in ("", None, "safe", "unclassified")}, {"unclassified"})


class SearchQueryTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelId("test-model", "weights")

    def test_saved_query_round_trips_without_loss(self):
        query = SearchQuery((1., -.1, 2e-3, 1e-30), False, self.model)
        text = query.to_text()
        self.assertEqual(parse_search_query(text), query)
        self.assertEqual(parse_search_query(f"  {text}\n", self.model), query)
        self.assertEqual(json.loads(text)["version"], 1)

    def test_query_saved_for_another_model_is_rejected_when_a_model_is_expected(self):
        text = SearchQuery((1., 0.), True, ModelId("other", "")).to_text()
        self.assertEqual(parse_search_query(text).model_id, ModelId("other", ""))
        with self.assertRaisesRegex(InvalidSearchQuery, "different model.*other .no pretrained tag."):
            parse_search_query(text, self.model)

    def test_parser_rejects_invalid_schema_and_numbers(self):
        valid = json.loads(SearchQuery((1., 0.), False, self.model).to_text())
        for change in (
            {"version": 2}, {"version": True}, {"version": None},
            {"mean_centering": "false"}, {"mean_centering": None},
            {"model": None}, {"model": {"model_name": "test"}}, {"model": {"model_name": " ", "pretrained": ""}},
            {"vector": []}, {"vector": [[1., 2.]]}, {"vector": [True]}, {"vector": "1, 2"},
            {"vector": [float("nan")]}, {"vector": [1e100]}, {"vector": [10 ** 400]},
        ):
            with self.subTest(change=change), self.assertRaises(InvalidSearchQuery):
                parse_search_query(json.dumps({**valid, **change}))

    def test_parser_rejects_text_that_is_not_a_saved_query(self):
        for text in (None, 1, "", "  ", "garbage", "[1., -.1, 2e-3]", "[[1, 2]]", "1, 2", "null", "{", "[" * 100_000):
            with self.subTest(text=str(text)[:20]), self.assertRaises(InvalidSearchQuery):
                parse_search_query(text)


if __name__ == "__main__":
    unittest.main()
