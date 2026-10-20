import json
import unittest
from unittest.mock import Mock

from app.domain.domain_object import Image, ImageId, ImageItem, ModelId, ModelItem, SearchResult, SearchResults
from app.domain.errors import ImageUnavailableError, ResourceLimitError
from app.presentation.controller import THUMBNAIL_MAX_AGE_SECONDS, create_app

MODEL = ModelId("Main", "weights")
MODEL_QUERY = {"model_name": "Main", "pretrained": "weights"}


class ControllerCatalogTests(unittest.TestCase):
    def setUp(self):
        self.usecase = Mock()
        self.usecase.get_image_items_by_page.return_value = [
            ImageItem(ImageId("a"), "dir/a.png", "red, cat", "general"),
        ]
        self.usecase.get_catalog_count.side_effect = lambda model, ratings=None: 6 if ratings is None else 2
        self.client = create_app(self.usecase).test_client()

    def get_page(self, **query):
        return self.client.get("/image_item", query_string={**MODEL_QUERY, **query})

    def test_page_items_expose_only_the_fields_needed_to_list_and_filter(self):
        response = self.get_page(page=2, size=60)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [
            {"id": "a", "name": "dir/a.png", "rating": "general"},
        ])
        self.usecase.get_image_items_by_page.assert_called_once_with(MODEL, 2, 60, None)

    def test_totals_are_counted_only_when_requested(self):
        response = self.get_page()
        self.assertNotIn("X-Catalog-Total", response.headers)
        self.usecase.get_catalog_count.assert_not_called()
        self.usecase.get_image_items_by_page.assert_called_once_with(MODEL, 0, 60, None)

        response = self.get_page(include_total=1)
        self.assertEqual((response.headers["X-Catalog-Total"], response.headers["X-Matching-Total"]), ("6", "6"))
        self.usecase.get_catalog_count.assert_called_once_with(MODEL)

    def test_filtered_pages_report_global_and_matching_totals(self):
        response = self.get_page(ratings=json.dumps(["general"]), include_total=1)
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.headers["X-Catalog-Total"], response.headers["X-Matching-Total"]), ("6", "2"))
        self.usecase.get_image_items_by_page.assert_called_once_with(MODEL, 0, 60, ["general"])

    def test_known_empty_filter_returns_totals_without_a_redundant_page_scan(self):
        self.usecase.get_catalog_count.side_effect = lambda model, ratings=None: 7099334 if ratings is None else 0
        response = self.get_page(ratings='["unclassified"]', include_total=1)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])
        self.assertEqual(response.headers["X-Catalog-Total"], "7099334")
        self.assertEqual(response.headers["X-Matching-Total"], "0")
        self.usecase.get_image_items_by_page.assert_not_called()

    def test_pagination_model_and_ratings_are_validated_before_the_usecase(self):
        self.assertEqual(self.get_page(page=2, size=240).status_code, 200)
        self.usecase.reset_mock()
        for query in (
            {"page": -1}, {"page": "x"}, {"page": 1.5}, {"size": 0}, {"size": -1}, {"size": 241}, {"size": "x"},
            {"model_name": ""}, {"ratings": "general"}, {"ratings": "{}"}, {"ratings": "null"},
            {"ratings": '["unknown"]'}, {"ratings": "[null]"}, {"ratings": "[{}]"},
            {"ratings": json.dumps(["general"] * 6)}, {"ratings": "[" * 200},
        ):
            with self.subTest(query=query):
                response = self.get_page(**query)
                self.assertEqual(response.status_code, 400, response.data)
                self.assertEqual(response.get_json()["error_code"], "invalid_input")
        self.assertEqual(self.client.get("/image_item").status_code, 400)
        self.assertEqual(self.usecase.mock_calls, [])

    def test_models_are_listed_with_their_identifiers(self):
        self.usecase.get_all_model.return_value = [
            ModelItem(ModelId("Qwen/Qwen3-VL-Embedding-2B", ""), "Qwen"), ModelItem(MODEL, "Main-weights"),
        ]
        self.assertEqual(self.client.get("/model_item").get_json(), [
            {"model_name": "Qwen/Qwen3-VL-Embedding-2B", "pretrained": ""},
            {"model_name": "Main", "pretrained": "weights"},
        ])

    def test_image_metadata_uses_the_requested_model_and_reports_unset_quality(self):
        self.usecase.get_image_metadata.return_value = ImageItem(
            ImageId("image-001"), "a.png", "qwen tag", "general", "cluster-a", 7.5,
        )
        query = {"model_name": "Qwen/Qwen3-VL-Embedding-2B", "pretrained": ""}
        response = self.client.get("/image_meta/image-001", query_string=query)
        self.assertEqual(response.get_json(), {
            "tags": "qwen tag", "style_cluster": "cluster-a", "rating": "general", "aesthetic_quality": 7.5,
            "blake3": None,
        })
        self.usecase.get_image_metadata.assert_called_once_with(
            ModelId("Qwen/Qwen3-VL-Embedding-2B", ""), ImageId("image-001"),
        )
        self.usecase.get_image_metadata.return_value = ImageItem(ImageId("b"), "b.png", blake3="ab" * 32)
        metadata = self.client.get("/image_meta/b", query_string=query).get_json()
        self.assertEqual((metadata["aesthetic_quality"], metadata["blake3"]), (None, "ab" * 32))
        self.usecase.get_image_metadata.return_value = None
        self.assertEqual(self.client.get("/image_meta/missing", query_string=query).status_code, 404)
        self.assertEqual(self.client.get("/image_meta/b").status_code, 400)

    def test_search_results_include_the_rating_needed_for_filtering(self):
        self.usecase.search_random.return_value = SearchResults(
            [SearchResult(ImageItem(ImageId("a"), "a.png", "tag", "explicit", "style_1", 5.0), 0.25)], "query",
        )
        response = self.client.post("/search/random", json={"params": MODEL_QUERY})
        self.assertEqual(response.get_json(), {
            "list": [{"item": {"id": "a", "name": "a.png", "rating": "explicit"}, "score": 0.25}],
            "search_query": "query",
        })


class ControllerImageTests(unittest.TestCase):
    def setUp(self):
        self.usecase = Mock()
        self.client = create_app(self.usecase).test_client()

    def test_thumbnails_are_cacheable_and_originals_are_always_fetched(self):
        self.usecase.get_small_image.return_value = Image(b"small", "image/webp", "a.webp")
        self.usecase.get_image.return_value = Image(b"original", "image/png", "a.png")

        small = self.client.get("/image/a/small")
        self.assertEqual((small.status_code, small.data, small.content_type), (200, b"small", "image/webp"))
        self.assertTrue(small.cache_control.private)
        self.assertEqual(small.cache_control.max_age, THUMBNAIL_MAX_AGE_SECONDS)
        self.usecase.get_small_image.assert_called_once_with(ImageId("a"))

        original = self.client.get("/image/a/original")
        self.assertEqual((original.data, original.content_type), (b"original", "image/png"))
        self.assertNotIn("Cache-Control", original.headers)
        self.assertEqual(original.headers["X-Content-Type-Options"], "nosniff")

    def test_unavailable_and_oversized_images_have_distinct_statuses(self):
        for error, status in ((ImageUnavailableError("unknown"), 404), (ResourceLimitError("too large"), 413)):
            self.usecase.get_small_image.side_effect = error
            self.usecase.get_image.side_effect = error
            for variant in ("small", "original"):
                with self.subTest(error=type(error).__name__, variant=variant):
                    response = self.client.get(f"/image/unknown/{variant}")
                    self.assertEqual(response.status_code, status)
                    self.assertNotIn("max-age", response.headers.get("Cache-Control", ""))


if __name__ == "__main__":
    unittest.main()
