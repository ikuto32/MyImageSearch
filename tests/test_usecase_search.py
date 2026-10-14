from io import BytesIO
import json
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np
from PIL import Image as PILImage

from app.application.search import SearchOptions
from app.application.text_matching import SearchPatternError
from app.application.usecase import WARMUP_TEXT, Usecase
from app.domain import limits
from app.domain.domain_object import Image, ImageId, ModelId
from app.domain.errors import ImageUnavailableError, ResourceLimitError, SearchInputError
from app.domain.search_query import InvalidSearchQuery, SearchQuery, parse_search_query
from helpers import make_search_index


def png_bytes(size=(1, 1)) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", size).save(buffer, format="PNG")
    return buffer.getvalue()


class UsecaseSearchTestCase(unittest.TestCase):
    def setUp(self):
        self.model = ModelId("test-model", "weights")
        faiss_index = faiss.IndexFlatIP(3)
        faiss_index.add(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0]], dtype=np.float32))
        self.search_index = make_search_index(
            faiss_index, ["0", "1", "2", "3"],
            mean_vector=np.array([.6, -.15, .1], dtype=np.float32),
            tags=["red cat", "blue dog", "red dog", ""],
            style_clusters=["style_1", "style_2", "style_12", ""],
            original=[3, 4, 5, 6], pony=[6, 5, 4, 3],
        )
        self.accessor = Mock()
        self.accessor.load_search_index.return_value = self.search_index
        self.backend = self.accessor.load_embedding_backend.return_value
        self.backend.encode_text.side_effect = lambda _: np.array([[.8, .6, 1e-9]], dtype=np.float32)
        self.backend.encode_image.side_effect = lambda _: np.array([[.72, .61, .23]], dtype=np.float32)
        self.image = Image(png_bytes(), "image/png")
        self.repository = Mock()
        self.repository.load_image.return_value = self.image
        self.usecase = Usecase(self.repository, self.accessor)
        self.options = SearchOptions(result_size=4)

    @staticmethod
    def ids(results):
        return [result.item.id for result in results.items]

    def assert_same_results(self, first, replay):
        self.assertEqual(self.ids(first), self.ids(replay))
        self.assertEqual([r.score for r in first.items], [r.score for r in replay.items])

    def replay(self, results, options=None):
        return self.usecase.search_query(
            self.model, parse_search_query(results.search_query), options or self.options,
        )


class SavedQueryReplayTests(UsecaseSearchTestCase):
    def test_text_replay_preserves_noncentered_ranking_and_exact_scores(self):
        results = self.usecase.search_text(self.model, "cat", self.options)
        data = json.loads(results.search_query)
        self.assertEqual(data["version"], 1)
        self.assertFalse(data["mean_centering"])
        self.assertEqual(data["model"], {"model_name": "test-model", "pretrained": "weights"})
        self.assertNotEqual(data["vector"][2], 0)
        self.assert_same_results(results, self.replay(results))
        centered = self.usecase.search_query(
            self.model, SearchQuery(tuple(data["vector"]), True, self.model), self.options,
        )
        self.assertNotEqual(self.ids(results), self.ids(centered))

    def test_image_upload_and_random_queries_replay_exactly(self):
        searches = [
            self.usecase.search_image(self.model, [ImageId("0"), ImageId("1")], self.options),
            self.usecase.search_upload_image(self.model, self.image, self.options),
            self.usecase.search_random(self.model, self.options),
        ]
        for results in searches:
            with self.subTest(query=results.search_query):
                self.assertTrue(json.loads(results.search_query)["mean_centering"])
                self.assert_same_results(results, self.replay(results))
                self.assert_same_results(results, self.replay(self.replay(results)))

    def test_add_text_preserves_source_mode_and_result_can_be_replayed(self):
        sources = [
            self.usecase.search_text(self.model, "cat", self.options),
            self.usecase.search_random(self.model, self.options),
        ]
        for source in sources:
            query = parse_search_query(source.search_query)
            unchanged = self.usecase.add_text_features(self.model, "cat", query, 0, self.options)
            self.assert_same_results(source, unchanged)
            changed = self.usecase.add_text_features(self.model, "cat", query, .5, self.options)
            self.assert_same_results(changed, self.replay(changed))
            self.assertEqual(parse_search_query(changed.search_query).mean_centering, query.mean_centering)

    def test_same_aesthetic_options_preserve_replay_scores(self):
        options = SearchOptions(.3, (3, 5), "pony", 3)
        results = self.usecase.search_text(self.model, "cat", options)
        self.assertTrue(results.items)
        self.assertTrue(set(self.ids(results)) <= {"1", "2", "3"})
        self.assert_same_results(results, self.replay(results, options))

    def test_zero_strength_refinement_does_not_load_embedding_model(self):
        query = SearchQuery((.8, .6, 0.), False, self.model)
        self.usecase.add_text_features(self.model, "cat", query, 0, self.options)
        self.accessor.load_embedding_backend.assert_not_called()

    def test_refinement_strength_is_independent_of_source_norm(self):
        for centered in (False, True):
            sources = [SearchQuery((scale, 0., 0.), centered, self.model) for scale in (1., 20.)]
            refined = [self.usecase.add_text_features(self.model, "cat", query, .5, self.options) for query in sources]
            self.assert_same_results(*refined)
            self.assertEqual(
                parse_search_query(refined[0].search_query).vector,
                parse_search_query(refined[1].search_query).vector,
            )
            for query in sources:
                original = self.usecase.search_query(self.model, query, self.options)
                unchanged = self.usecase.add_text_features(self.model, "cat", query, 0, self.options)
                self.assert_same_results(original, unchanged)
                self.assertEqual(query.vector, parse_search_query(unchanged.search_query).vector)

    def test_model_mismatch_is_rejected_before_index_or_backend_load(self):
        query = SearchQuery((1., 0., 0.), False, ModelId("other", ""))
        with self.assertRaisesRegex(InvalidSearchQuery, "different model"):
            self.usecase.search_query(self.model, query, self.options)
        with self.assertRaisesRegex(InvalidSearchQuery, "different model"):
            self.usecase.add_text_features(self.model, "cat", query, 1, self.options)
        self.accessor.load_search_index.assert_not_called()
        self.accessor.load_embedding_backend.assert_not_called()

    def test_saved_query_of_another_dimension_is_an_invalid_query(self):
        query = SearchQuery((1., 2.), False, self.model)
        with self.assertRaisesRegex(InvalidSearchQuery, "dimension mismatch"):
            self.usecase.search_query(self.model, query, self.options)
        with self.assertRaisesRegex(InvalidSearchQuery, "dimension mismatch"):
            self.usecase.add_text_features(self.model, "cat", query, 1, self.options)
        self.accessor.load_embedding_backend.assert_not_called()

    def test_embedding_of_another_dimension_is_not_blamed_on_the_user(self):
        self.backend.encode_text.side_effect = lambda _: np.zeros((1, 2), dtype=np.float32)
        with self.assertRaises(ValueError) as raised:
            self.usecase.search_text(self.model, "cat", self.options)
        self.assertNotIsInstance(raised.exception, (InvalidSearchQuery, SearchInputError))


class ImageSearchTests(UsecaseSearchTestCase):
    def test_multiple_images_search_with_the_mean_of_their_normalized_features(self):
        features = iter([[2., 0., 0.], [0., 4., 0.]])
        self.backend.encode_image.side_effect = lambda _: np.array([next(features)], dtype=np.float32)
        results = self.usecase.search_image(self.model, [ImageId("0"), ImageId("1")], self.options)
        self.assertEqual(parse_search_query(results.search_query).vector, (.5, .5, 0.))

    def test_unavailable_selected_image_is_a_search_input_error_without_loading_backend(self):
        self.repository.load_image.side_effect = ImageUnavailableError("not found")
        with self.assertRaisesRegex(SearchInputError, "missing"):
            self.usecase.search_image(self.model, [ImageId("missing")], self.options)
        self.accessor.load_embedding_backend.assert_not_called()

    def test_image_limits_are_checked_before_decode_and_backend(self):
        self.repository.load_image.return_value = Image(png_bytes((12, 10)), "image/png")
        for limit, value in (("MAX_IMAGE_PIXELS", 119), ("MAX_IMAGE_BYTES", 1)):
            with self.subTest(limit=limit), patch.object(limits, limit, value):
                with self.assertRaises(ResourceLimitError):
                    self.usecase.search_image(self.model, [ImageId("0")], self.options)
        self.accessor.load_embedding_backend.assert_not_called()

    def test_invalid_image_bytes_are_reported_as_search_input(self):
        with self.assertRaises(SearchInputError):
            self.usecase.search_upload_image(self.model, Image(b"invalid", "image/png"), self.options)
        self.accessor.load_embedding_backend.assert_not_called()

    def test_upload_search_applies_the_aesthetic_options(self):
        results = self.usecase.search_upload_image(self.model, self.image, SearchOptions(0, (5, 10), "original", 4))
        self.assertEqual(sorted(self.ids(results)), ["2", "3"])


class MetadataSearchTests(UsecaseSearchTestCase):
    def test_name_tags_and_style_cluster_are_searched_in_their_own_column(self):
        for field, text, expected in (
            ("name", "1", ["1"]), ("tags", "red", ["0", "2"]), ("tags", "dog", ["1", "2"]),
            ("style_cluster", "style_1", ["0", "2"]), ("tags", "missing", []),
        ):
            with self.subTest(field=field, text=text):
                results = self.usecase.search_metadata(self.model, field, text, False, self.options)
                self.assertEqual(self.ids(results), expected)
                self.assertTrue(all(result.score == 1.0 for result in results.items))
                self.assertEqual(results.search_query, "")
        self.accessor.load_embedding_backend.assert_not_called()

    def test_regular_expressions_and_result_limit(self):
        results = self.usecase.search_metadata(self.model, "style_cluster", r"^style_\d$", True, self.options)
        self.assertEqual(self.ids(results), ["0", "1"])
        limited = self.usecase.search_metadata(self.model, "name", r"\d", True, SearchOptions(result_size=2))
        self.assertEqual(self.ids(limited), ["0", "1"])
        with self.assertRaises(SearchPatternError):
            self.usecase.search_metadata(self.model, "name", "[", True, self.options)

    def test_aesthetic_options_filter_and_order_matches(self):
        options = SearchOptions(1, (4, 10), "original", 4)
        results = self.usecase.search_metadata(self.model, "tags", "dog", False, options)
        self.assertEqual([(r.item.id, r.score) for r in results.items], [("2", 5.0), ("1", 4.0)])
        pony = SearchOptions(1, (4, 10), "pony", 4)
        results = self.usecase.search_metadata(self.model, "tags", "dog", False, pony)
        self.assertEqual([(r.item.id, r.score) for r in results.items], [("1", 5.0), ("2", 4.0)])

    def test_results_are_announced_to_the_repository_that_will_serve_their_images(self):
        for search in (
            lambda: self.usecase.search_text(self.model, "cat", self.options),
            lambda: self.usecase.search_metadata(self.model, "tags", "dog", False, self.options),
        ):
            results = search()
            (items,), _ = self.repository.remember_image_items.call_args
            self.assertEqual(list(items), [result.item for result in results.items])
            self.assertTrue(results.items)


class WarmupTests(UsecaseSearchTestCase):
    def test_warmup_runs_one_minimal_text_search(self):
        self.usecase.warmup_search(self.model)
        self.accessor.load_embedding_backend.assert_called_once_with(self.model)
        self.backend.encode_text.assert_called_once_with(WARMUP_TEXT)
        self.accessor.load_search_index.assert_called_once_with(self.model)
        # Only one result is ranked.
        (items,), _ = self.repository.remember_image_items.call_args
        self.assertEqual(len(list(items)), 1)


if __name__ == "__main__":
    unittest.main()
