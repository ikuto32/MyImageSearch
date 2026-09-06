import json
import logging
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np
from PIL import Image as PILImage

from app.application.usecase import Usecase
from app.domain.domain_object import Image, ImageId, ImageItem, ImageName, ModelId, UploadImage, UploadText
from app.domain.errors import ResourceLimitError, SearchInputError
from app.domain.search_query import InvalidSearchQuery, SearchQuery, parse_search_query
from app.presentation import controller


class SavedSearchQueryTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelId("test-model", "weights")
        self.index = faiss.IndexFlatIP(3)
        self.index.add(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0]], dtype=np.float32))
        self.items = [ImageItem(ImageId(str(i)), ImageName(str(i)), aesthetic_quality=i + 3) for i in range(4)]
        self.accessor = Mock()
        self.accessor.load_index_with_metadata.return_value = (self.index, self.items)
        self.accessor.get_mean_meta_vector.return_value = np.array([.6, -.15, .1], dtype=np.float32)
        self.backend = self.accessor.load_embedding_backend.return_value
        self.backend.encode_text.side_effect = lambda _: np.array([[.8, .6, 1e-9]], dtype=np.float32)
        self.backend.encode_image.side_effect = lambda _: np.array([[.72, .61, .23]], dtype=np.float32)
        buffer = BytesIO()
        PILImage.new("RGB", (1, 1)).save(buffer, format="PNG")
        self.image = Image(buffer.getvalue(), "image/png")
        self.repository = Mock()
        self.repository.load_image.return_value = self.image
        self.usecase = Usecase.__new__(Usecase)
        self.usecase._logger = logging.getLogger(__name__)
        self.usecase._accessor = self.accessor
        self.usecase._repository = self.repository
        self.usecase._id_to_image_items = {}
        self.options = (0, 0, 10, "original", 4)

    def assert_same_results(self, first, replay):
        self.assertEqual([item.item.id for item in first.list], [item.item.id for item in replay.list])
        self.assertEqual([item.score.score for item in first.list], [item.score.score for item in replay.list])

    def replay(self, result, options=None):
        return self.usecase.search_query(self.model, result.search_query, *(options or self.options))

    def test_text_replay_preserves_noncentered_ranking_and_exact_scores(self):
        result = self.usecase.search_text(self.model, UploadText("cat"), *self.options)
        data = json.loads(result.search_query)
        self.assertEqual(data["version"], 1)
        self.assertFalse(data["mean_centering"])
        self.assertEqual(data["model"], {"model_name": "test-model", "pretrained": "weights"})
        self.assertNotEqual(data["vector"][2], 0)
        self.assert_same_results(result, self.replay(result))
        legacy = self.usecase.search_query(self.model, json.dumps(data["vector"]), *self.options)
        self.assertNotEqual([r.item.id for r in result.list], [r.item.id for r in legacy.list])

    def test_image_upload_and_random_queries_replay_exactly(self):
        results = [
            self.usecase.search_image(self.model, [ImageId("0"), ImageId("1")], *self.options),
            self.usecase.search_upload_image(self.model, UploadImage(self.image.binary, self.image.content_type), 4),
            self.usecase.search_random(self.model, *self.options),
        ]
        for result in results:
            with self.subTest(query=result.search_query):
                self.assertTrue(json.loads(result.search_query)["mean_centering"])
                self.assert_same_results(result, self.replay(result))
                self.assert_same_results(result, self.replay(self.replay(result)))

    def test_add_text_preserves_source_mode_and_result_can_be_replayed(self):
        sources = [
            self.usecase.search_text(self.model, UploadText("cat"), *self.options),
            self.usecase.search_random(self.model, *self.options),
        ]
        for source in sources:
            unchanged = self.usecase.add_text_features(self.model, UploadText("cat"), source.search_query, 0, *self.options)
            self.assert_same_results(source, unchanged)
            changed = self.usecase.add_text_features(self.model, UploadText("cat"), source.search_query, .5, *self.options)
            self.assert_same_results(changed, self.replay(changed))
            self.assertEqual(json.loads(changed.search_query)["mean_centering"], json.loads(source.search_query)["mean_centering"])

    def test_same_aesthetic_options_preserve_replay_scores(self):
        options = (.3, 3, 5, "pony", 3)
        result = self.usecase.search_text(self.model, UploadText("cat"), *options)
        self.assert_same_results(result, self.replay(result, options))

    def test_zero_strength_refinement_does_not_load_embedding_model(self):
        query = SearchQuery((.8, .6, 0.), False, self.model).to_text()
        self.usecase.add_text_features(self.model, UploadText("cat"), query, 0, *self.options)
        self.accessor.load_embedding_backend.assert_not_called()

    def test_refinement_strength_is_independent_of_source_norm(self):
        for centered in (False, True):
            sources = [SearchQuery((scale, 0., 0.), centered, self.model).to_text() for scale in (1., 20.)]
            results = [self.usecase.add_text_features(self.model, UploadText("cat"), query, .5, *self.options) for query in sources]
            self.assert_same_results(*results)
            self.assertEqual(parse_search_query(results[0].search_query).vector, parse_search_query(results[1].search_query).vector)
            for query in sources:
                original = self.usecase.search_query(self.model, query, *self.options)
                unchanged = self.usecase.add_text_features(self.model, UploadText("cat"), query, 0, *self.options)
                self.assert_same_results(original, unchanged)
                self.assertEqual(parse_search_query(query).vector, parse_search_query(unchanged.search_query).vector)

    def test_model_mismatch_is_rejected_before_index_or_backend_load(self):
        query = SearchQuery((1., 0., 0.), False, ModelId("other", "")).to_text()
        for method, args in (
            (self.usecase.search_query, (self.model, query, *self.options)),
            (self.usecase.add_text_features, (self.model, UploadText("cat"), query, 1, *self.options)),
        ):
            with self.assertRaisesRegex(InvalidSearchQuery, "different model"):
                method(*args)
        self.accessor.load_index_with_metadata.assert_not_called()
        self.accessor.load_embedding_backend.assert_not_called()

    def test_api_preserves_query_state_and_returns_400_for_invalid_dimensions(self):
        result = self.usecase.search_text(self.model, UploadText("cat"), *self.options)
        with patch.object(controller, "usecase", self.usecase):
            client = controller.app.test_client()
            params = {"model_name": self.model.model_name, "pretrained": self.model.pretrained, "search_query": result.search_query, "result_size": 4}
            response = client.post("/search/query", json={"params": params})
            self.assertEqual(response.status_code, 200)
            with controller.app.app_context():
                expected = controller.from_result_to_json(result).get_json()["list"]
            self.assertEqual(response.get_json()["list"], expected)
            for query in ("[1, 2]", SearchQuery((1., 2.), False, self.model).to_text()):
                response = client.post("/search/query", json={"params": {**params, "search_query": query}})
                self.assertEqual(response.status_code, 400)
                self.assertIn("dimension mismatch", response.get_json()["error"])

    def test_saved_query_parser_rejects_invalid_schema_and_numbers(self):
        valid = json.loads(SearchQuery((1., 0.), False, self.model).to_text())
        for change in (
            {"version": 2}, {"version": True}, {"version": None},
            {"mean_centering": "false"}, {"mean_centering": None},
            {"model": None}, {"model": {"model_name": "test"}},
            {"vector": []}, {"vector": [[1., 2.]]}, {"vector": [True]},
            {"vector": [float("nan")]}, {"vector": [1e100]},
        ):
            with self.subTest(change=change), self.assertRaises(InvalidSearchQuery):
                parse_search_query(json.dumps({**valid, **change}))

    def test_legacy_numpy_vectors_remain_centered(self):
        for text in ("[[1., -.1, 2e-3]]", "[1., -.1, 2e-3]", "1., -.1, 2e-3"):
            parsed = parse_search_query(text)
            self.assertEqual(parsed.vector, (1., -.1, .002))
            self.assertTrue(parsed.mean_centering)
            self.assertIsNone(parsed.model_id)

    def test_unknown_selected_image_returns_400_without_loading_backend(self):
        self.repository.load_image.side_effect = ValueError("not found")
        with patch.object(controller, "usecase", self.usecase):
            response = controller.app.test_client().post("/search/image", json={"params": {
                "model_name": self.model.model_name, "pretrained": self.model.pretrained,
                "id": ["missing"],
            }})
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing", response.get_json()["error"])
        self.accessor.load_embedding_backend.assert_not_called()

    def test_selected_image_pixel_limit_is_checked_before_decode_and_backend(self):
        decoded = Mock(width=8001, height=8000)
        with patch.object(Image, "to_ptl_image", return_value=decoded):
            with self.assertRaises(ResourceLimitError):
                self.usecase.search_image(self.model, [ImageId("0")], *self.options)
        decoded.load.assert_not_called()
        decoded.close.assert_called_once()
        self.accessor.load_embedding_backend.assert_not_called()

    def test_invalid_image_bytes_are_reported_as_search_input(self):
        with self.assertRaises(SearchInputError):
            self.usecase.search_upload_image(self.model, UploadImage(b"invalid", "image/png"))
        self.accessor.load_embedding_backend.assert_not_called()

    def test_metadata_lookups_use_only_requested_ids_without_search_index(self):
        self.accessor.load_image_metadata.return_value = self.items[:1]
        self.usecase.get_image_metadata(self.model, ImageId("0"))
        self.accessor.load_image_metadata.assert_called_with(self.model, [ImageId("0")])
        self.usecase.get_rating_list(self.model, [ImageId("0")])
        self.accessor.load_image_metadata.assert_called_with(self.model, [ImageId("0")])
        self.accessor.load_index_with_metadata.assert_not_called()

    def test_zip_images_are_loaded_on_demand(self):
        expected = object()

        def consume(images):
            self.repository.load_image.assert_not_called()
            next(images)
            self.assertEqual(self.repository.load_image.call_count, 1)
            next(images)
            self.assertEqual(self.repository.load_image.call_count, 2)
            return expected

        self.repository.create_zip_from_images.side_effect = consume
        self.assertIs(self.usecase.get_images_zip(["0", "1"]), expected)


if __name__ == "__main__":
    unittest.main()
