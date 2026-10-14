import unittest

import faiss
import numpy as np

from app.application.search import (
    SearchOptions,
    apply_aesthetics,
    check_dimension,
    nearest_neighbors,
    rank_results,
)
from app.application.search_index import SearchIndex
from app.domain.search_query import InvalidSearchQuery
from helpers import make_search_index


def flat_index(vectors) -> faiss.Index:
    vectors = np.array(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def options(size=10, beta=0.0, low=0.0, high=10.0, model="original") -> SearchOptions:
    return SearchOptions(beta, (low, high), model, size)


class UniterableNames(list):
    """検索は全件走査せず、FAISS が返した行番号のみを参照する。"""

    def __iter__(self):
        raise AssertionError("Search must not traverse or sort all image metadata")


class NearestNeighborTests(unittest.TestCase):
    def search(self, index, features, **kwargs):
        positions, scores = nearest_neighbors(
            index, np.array([features], dtype=np.float32),
            mean_centering=kwargs.pop("mean_centering", False), options=options(**kwargs),
        )
        return [index.names[position] for position in positions], scores.tolist()

    def test_faiss_row_numbers_select_metadata_without_scanning_it(self):
        names = UniterableNames(["z.png", "a.png"])
        unrated = np.full(2, np.nan)
        index = SearchIndex(
            index=flat_index([[1, 0], [0, 1]]), mean_vector=None, ids=names, names=names,
            tags=["", ""], ratings=["", ""], style_clusters=["", ""],
            aesthetic_scores={"original": unrated, "pony": unrated}, names_sorted=True,
        )
        positions, scores = nearest_neighbors(
            index, np.array([[1, 0]], dtype=np.float32), mean_centering=False, options=options(size=1),
        )
        results = rank_results(index, positions, scores, 1)
        self.assertEqual([result.item.id for result in results], ["z.png"])
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_empty_index_does_not_search_with_zero_neighbors(self):
        class EmptyIndex:
            d = 2
            ntotal = 0

            def search(self, *_args, **_kwargs):
                raise AssertionError("FAISS must not be called with k=0")

        self.assertEqual(self.search(make_search_index(EmptyIndex(), []), [1, 0]), ([], []))

    def test_result_limit_is_clamped_to_index_size(self):
        index = make_search_index(flat_index([[1, 0], [0, 1]]), ["a.png", "b.png"])
        self.assertEqual(self.search(index, [1, 0], size=1000)[0], ["a.png", "b.png"])

    def test_mismatched_index_and_metadata_fail_instead_of_mislabeling_results(self):
        with self.assertRaisesRegex(ValueError, "2 vectors.*1 images"):
            make_search_index(flat_index([[1, 0], [0, 1]]), ["a.png"])

    def test_query_is_normalized_then_centered_without_renormalizing(self):
        index = make_search_index(
            flat_index([[1, 0], [0, 1]]), ["a.png", "b.png"], mean_vector=np.array([0.5, 0.25], dtype=np.float32),
        )
        names, scores = self.search(index, [4, 0], mean_centering=True)
        self.assertEqual(names, ["a.png", "b.png"])
        np.testing.assert_allclose(scores, [0.5, -0.25])
        self.assertEqual(self.search(index, [4, 0], mean_centering=False)[1], [1.0, 0.0])

    def test_uncentered_index_ignores_centering_request(self):
        index = make_search_index(flat_index([[1, 0]]), ["a.png"])
        self.assertEqual(self.search(index, [4, 0], mean_centering=True)[1], [1.0])

    def test_dimension_mismatch_is_reported_with_the_requested_error_type(self):
        index = make_search_index(flat_index([[1, 0]]), ["a.png"])
        with self.assertRaisesRegex(ValueError, "query has 3 dimensions.*expects 2"):
            self.search(index, [1, 0, 0])
        with self.assertRaises(InvalidSearchQuery):
            check_dimension(index, np.zeros((1, 3), dtype=np.float32), error_type=InvalidSearchQuery)

    def test_aesthetic_range_is_applied_before_neighbor_limit(self):
        index = make_search_index(
            flat_index([[1, 0], [.9, .1]]), ["excluded.png", "included.png"], original=[1, 7],
        )
        self.assertEqual(self.search(index, [1, 0], size=1, low=6, high=8)[0], ["included.png"])

    def test_aesthetic_range_uses_the_selected_model_scores(self):
        index = make_search_index(
            flat_index([[1, 0], [.9, .1]]), ["a.png", "b.png"], original=[1, 7], pony=[7, 1],
        )
        self.assertEqual(self.search(index, [1, 0], low=6, high=8)[0], ["b.png"])
        self.assertEqual(self.search(index, [1, 0], low=6, high=8, model="pony")[0], ["a.png"])

    def test_bitmap_handles_ivf_pq_without_overfetch(self):
        rng = np.random.default_rng(4)
        ivf = faiss.IndexIVFPQ(faiss.IndexFlatIP(4), 4, 2, 1, 2, faiss.METRIC_INNER_PRODUCT)
        ivf.train(rng.normal(size=(200, 4)).astype(np.float32))
        ivf.add(np.array([[1, 0, 0, 0], [.9, .1, 0, 0]], dtype=np.float32))
        index = make_search_index(ivf, ["excluded.png", "included.png"], original=[1, 7])
        self.assertEqual(self.search(index, [1, 0, 0, 0], size=1, low=6, high=8)[0], ["included.png"])
        # Coarse PQ codes do not preserve the exact order; both rows must still be found.
        self.assertEqual(sorted(self.search(index, [1, 0, 0, 0], size=2)[0]), ["excluded.png", "included.png"])

    def test_every_inverted_list_is_probed_without_changing_the_index(self):
        # One vector per inverted list: probing a single list, the FAISS default, would find one image.
        axes = np.eye(4, dtype=np.float32)
        quantizer = faiss.IndexFlatIP(4)
        quantizer.add(axes)
        ivf = faiss.IndexIVFFlat(quantizer, 4, 4, faiss.METRIC_INNER_PRODUCT)
        ivf.train(axes)
        ivf.add(axes)
        index = make_search_index(ivf, ["a.png", "b.png", "c.png", "d.png"], original=[1, 7, 7, 7])
        query = [1, .5, .25, .1]
        self.assertEqual(self.search(index, query)[0], ["a.png", "b.png", "c.png", "d.png"])
        self.assertEqual(self.search(index, query, low=6, high=8)[0], ["b.png", "c.png", "d.png"])
        self.assertEqual(ivf.nprobe, 1)

    def test_aesthetic_bitmap_uses_full_score_precision_at_boundaries(self):
        index = make_search_index(
            flat_index([[1, 0], [.9, .1]]), ["just_outside.png", "boundary.png"], original=[4.90000001, 4.9],
        )
        self.assertEqual(self.search(index, [1, 0], size=1, high=4.9)[0], ["boundary.png"])

    def test_no_eligible_images_skip_faiss(self):
        class Index:
            d = 2
            ntotal = 1

            def search(self, *_args, **_kwargs):
                raise AssertionError("No eligible images should skip FAISS")

        index = make_search_index(Index(), ["unrated.png"])
        self.assertEqual(self.search(index, [1, 0], size=1, low=6, high=8), ([], []))


class AestheticScoringTests(unittest.TestCase):
    def setUp(self):
        self.index = make_search_index(
            flat_index(np.eye(4)), ["inside", "outside", "missing", "upper"], original=[6, 1, np.nan, 8],
        )
        self.positions = np.arange(4)
        self.scores = np.array([-0.2, 0.8, 0.9, 0.4])

    def apply(self, **kwargs):
        positions, scores = apply_aesthetics(self.index, self.positions, self.scores, options(**kwargs))
        return [self.index.names[position] for position in positions], scores.tolist()

    def test_range_excludes_outside_and_unrated_images_and_keeps_scores(self):
        self.assertEqual(self.apply(low=6), (["inside", "upper"], [-0.2, 0.4]))

    def test_range_includes_boundaries_and_weights_scores(self):
        names, scores = self.apply(beta=0.5, low=6, high=8)
        self.assertEqual(names, ["inside", "upper"])
        np.testing.assert_allclose(scores, [-0.2 * 0.75 + 6 * 0.5, 0.4 * 0.75 + 8 * 0.5])

    def test_filter_and_weighting_use_the_scores_of_the_selected_model(self):
        index = make_search_index(flat_index(np.eye(2)), ["a", "b"], original=[2, 8], pony=[8, 2])
        for model, expected in (("original", ([1], [8.0])), ("pony", ([0], [8.0]))):
            with self.subTest(model=model):
                positions, scores = apply_aesthetics(
                    index, np.arange(2), np.zeros(2), options(beta=1, low=5, model=model),
                )
                self.assertEqual((positions.tolist(), scores.tolist()), expected)

    def test_weighting_alone_also_excludes_unrated_images(self):
        self.assertEqual(self.apply(beta=-1)[0], ["inside", "outside", "upper"])

    def test_disabled_filter_preserves_unrated_items(self):
        positions, scores = apply_aesthetics(self.index, self.positions, self.scores, options())
        self.assertIs(positions, self.positions)
        self.assertIs(scores, self.scores)


class RankingTests(unittest.TestCase):
    NAMES = ["image10.png", "image02.png", "image01.png", "image00.png"]

    def ranked(self, names, scores, limit, **kwargs):
        index = make_search_index(flat_index(np.eye(len(names))), names, **kwargs)
        positions = np.arange(len(names))
        return [
            (result.item.name, result.score)
            for result in rank_results(index, positions, np.array(scores, dtype=np.float64), limit)
        ]

    def test_results_are_sorted_by_score_then_name_and_limited(self):
        expected = [("image01.png", 0.9), ("image02.png", 0.9), ("image10.png", 0.5)]
        self.assertEqual(self.ranked(self.NAMES, [0.5, 0.9, 0.9, 0.1], 3), expected)

    def test_path_ordered_rows_rank_identically_without_comparing_names(self):
        names = sorted(self.NAMES)
        scores = [0.1, 0.9, 0.9, 0.5]
        expected = [("image01.png", 0.9), ("image02.png", 0.9), ("image10.png", 0.5)]
        self.assertEqual(self.ranked(names, scores, 3), expected)
        self.assertEqual(self.ranked(names, scores, 3, names_sorted=False), expected)

    def test_result_items_carry_their_metadata(self):
        index = make_search_index(
            flat_index(np.eye(2)), ["a.png", "b.png"], ids=["id-a", "id-b"], tags=["red", ""],
            ratings=["general", ""], style_clusters=["style_1", ""], original=[7.5, np.nan],
        )
        first, second = (result.item for result in rank_results(index, np.arange(2), np.array([1.0, 0.5]), 2))
        self.assertEqual(
            (first.id, first.name, first.tags, first.rating, first.style_cluster, first.aesthetic_quality),
            ("id-a", "a.png", "red", "general", "style_1", 7.5),
        )
        self.assertEqual((second.id, second.name), ("id-b", "b.png"))
        self.assertIsNone(second.aesthetic_quality)


if __name__ == "__main__":
    unittest.main()
