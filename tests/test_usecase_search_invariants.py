import logging
import unittest

import faiss
import numpy as np

from app.application.usecase import Usecase
from app.domain.domain_object import (
    ImageId,
    ImageItem,
    ImageName,
    ModelId,
    ResultImageItem,
    Score,
)


class IndexOrderedItems(list):
    """検索は全件走査せず、FAISS が返した行番号のみを参照する。"""

    def __iter__(self):
        raise AssertionError("Search must not traverse or sort all image metadata")


def make_item(name, aesthetic_quality=None):
    return ImageItem(
        ImageId(name), ImageName(name), aesthetic_quality=aesthetic_quality
    )


class UsecaseSearchInvariantTests(unittest.TestCase):
    def setUp(self):
        self.usecase = Usecase.__new__(Usecase)
        self.usecase._logger = logging.getLogger(__name__)
        self.model_id = ModelId("test-model", "")

    def test_faiss_row_numbers_use_accessor_order_without_metadata_scan(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
        items = IndexOrderedItems([make_item("z.png"), make_item("a.png")])

        results = self.usecase.similarity_eval(
            items, index, np.array([[1, 0]], dtype=np.float32),
            result_size=1, mean_centering=False,
        )

        self.assertEqual([result.item.id.id for result in results], ["z.png"])
        self.assertAlmostEqual(results[0].score.score, 1.0)

    def test_empty_index_does_not_search_with_zero_neighbors(self):
        class EmptyIndex:
            d = 2
            ntotal = 0

            def search(self, *_args, **_kwargs):
                raise AssertionError("FAISS must not be called with k=0")

        results = self.usecase.similarity_eval(
            [], EmptyIndex(), np.array([[1, 0]], dtype=np.float32),
            result_size=10, mean_centering=False,
        )

        self.assertEqual(results, [])

    def test_result_limit_is_clamped_to_index_size(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
        items = IndexOrderedItems([make_item("a.png"), make_item("b.png")])

        results = self.usecase.similarity_eval(
            items, index, np.array([[1, 0]], dtype=np.float32),
            result_size=1000, mean_centering=False,
        )

        self.assertEqual([result.item.id.id for result in results], ["a.png", "b.png"])

    def test_mismatched_index_and_metadata_fail_instead_of_mislabeling_results(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "2 vectors.*1 images"):
            self.usecase.similarity_eval(
                [make_item("a.png")], index,
                np.array([[1, 0]], dtype=np.float32), mean_centering=False,
            )

    def test_aesthetic_range_excludes_outside_and_missing_scores(self):
        scores = [
            ResultImageItem(make_item("inside", 7), Score(-0.2)),
            ResultImageItem(make_item("outside", 1), Score(0.8)),
            ResultImageItem(make_item("missing"), Score(0.9)),
            ResultImageItem(make_item("nan", float("nan")), Score(0.9)),
        ]

        results = self.usecase.apply_aesthetic_quality_filter(
            self.model_id, scores, 0, 6, 10, "original"
        )

        self.assertEqual([result.item.id.id for result in results], ["inside"])
        self.assertEqual(results[0].score.score, -0.2)

    def test_aesthetic_range_includes_boundaries_and_retains_weighting(self):
        scores = [
            ResultImageItem(make_item("lower", 6), Score(0.8)),
            ResultImageItem(make_item("upper", 8), Score(0.4)),
        ]

        results = self.usecase.apply_aesthetic_quality_filter(
            self.model_id, scores, 0.5, 6, 8, "original"
        )

        self.assertEqual([result.item.id.id for result in results], ["lower", "upper"])
        self.assertAlmostEqual(results[0].score.score, 3.6)
        self.assertAlmostEqual(results[1].score.score, 4.3)

    def test_disabled_aesthetic_filter_preserves_unrated_items(self):
        scores = [ResultImageItem(make_item("unrated"), Score(0.8))]

        results = self.usecase.apply_aesthetic_quality_filter(
            self.model_id, scores, 0, 0, 10, "original"
        )

        self.assertIs(results, scores)

    def test_aesthetic_range_is_applied_before_neighbor_limit(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [.9, .1]], dtype=np.float32))
        items = [make_item("excluded.png", 1), make_item("included.png", 7)]
        results = self.usecase.similarity_eval(
            items, index, np.array([[1, 0]], dtype=np.float32),
            result_size=1, mean_centering=False, aesthetic_range=(6, 8),
        )
        self.assertEqual([result.item.id.id for result in results], ["included.png"])

    def test_bitmap_handles_ivf_pq_without_overfetch(self):
        rng = np.random.default_rng(4)
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(4), 4, 2, 1, 2, faiss.METRIC_INNER_PRODUCT)
        index.train(rng.normal(size=(200, 4)).astype(np.float32))
        index.add(np.array([[1, 0, 0, 0], [.9, .1, 0, 0]], dtype=np.float32))
        items = [make_item("excluded.png", 1), make_item("included.png", 7)]
        results = self.usecase.similarity_eval(
            items, index, np.array([[1, 0, 0, 0]], dtype=np.float32),
            result_size=1, mean_centering=False, aesthetic_range=(6, 8),
        )
        self.assertEqual([result.item.id.id for result in results], ["included.png"])

    def test_aesthetic_bitmap_uses_full_score_precision_at_boundaries(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [.9, .1]], dtype=np.float32))
        items = [make_item("just_outside.png", 4.90000001), make_item("boundary.png", 4.9)]
        results = self.usecase.similarity_eval(
            items, index, np.array([[1, 0]], dtype=np.float32),
            result_size=1, mean_centering=False, aesthetic_range=(0, 4.9),
        )
        self.assertEqual([result.item.id.id for result in results], ["boundary.png"])

    def test_empty_aesthetic_bitmap_does_not_call_faiss(self):
        class Index:
            d = 2
            ntotal = 1

            def search(self, *_args, **_kwargs):
                raise AssertionError("No eligible images should skip FAISS")

        results = self.usecase.similarity_eval(
            [make_item("unrated.png")], Index(), np.array([[1, 0]], dtype=np.float32),
            result_size=1, mean_centering=False, aesthetic_range=(6, 8),
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
