from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from indexer import features
from indexer.database import init_db
from indexer.dataset import DirImageIterable, safe_collate
from indexer.features import FeatureExtractor, MetadataModels, recalc_aesthetic_scores
from indexer.scoring import PonyAestheticScorer, StyleCluster
from helpers import save_image


def quiet(test: unittest.TestCase) -> None:
    for redirect in (redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
        redirect.__enter__()
        test.addCleanup(redirect.__exit__, None, None, None)


class SearchModel:
    """Embeds an image as (mean red, mean green) of its preprocessed tensor."""

    preprocess = None

    def __init__(self, dimension=2):
        self.dimension = dimension
        self.calls = 0

    def encode_image_with_internal(self, batch):
        self.calls += 1
        projected = batch.float().mean(dim=(2, 3))[:, :self.dimension]
        internal = torch.cat([projected, projected], dim=1)
        return internal, projected


class Aesthetic:
    def __init__(self):
        self.seen = None

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32)])

    def score_batch(self, batch):
        self.seen = batch
        return batch[:, 0]


class Pony:
    def __init__(self):
        self.seen = None

    def score_batch(self, batch):
        self.seen = batch
        return batch[:, 1].numpy() * 2


class Style:
    def __init__(self):
        self.seen = None

    def get_cluster_batch(self, batch):
        self.seen = batch
        return [7] * len(batch), np.zeros(len(batch))


def to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255)


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self):
        quiet(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.images = Path(temporary.name)
        self.paths = ["red.png", "green.png", "missing.png", "sub/yellow.png"]
        save_image(self.images / "red.png", color=(255, 0, 0))
        save_image(self.images / "green.png", color=(0, 255, 0))
        save_image(self.images / "sub/yellow.png", color=(255, 255, 0))
        (self.images / "red.png.tags.json").write_text(
            json.dumps({"rating": "general", "tags": ["red", "plain"]}), encoding="utf-8",
        )
        self.ids = [f"id-{index}" for index in range(len(self.paths))]
        self.connection = sqlite3.connect(":memory:")
        self.addCleanup(self.connection.close)
        init_db(self.connection)
        self.seen = np.zeros(len(self.paths), dtype=bool)

    def extractor(self, **overrides):
        options = dict(
            device="cpu", search_model=SearchModel(), search_model_label="fake:model",
            metadata_models=None, metadata_embedding=None, metadata_model_label=None,
            tagging_service=None, image_dir=str(self.images), dimension=None,
        )
        options.update(overrides)
        return FeatureExtractor(self.connection, **options)

    def batches(self, *, metadata_transform=None, tag_sizes=None, batch_size=2):
        dataset = DirImageIterable(
            str(self.images), self.paths, transform=to_tensor,
            tag_sizes=tag_sizes, metadata_transform=metadata_transform,
        )
        samples = list(dataset)
        return [safe_collate(samples[start:start + batch_size]) for start in range(0, len(samples), batch_size)]

    def run_extractor(self, extractor, batches):
        with mock.patch.object(features.tqdm, "tqdm", side_effect=lambda iterable, **_: iterable):
            extractor.run(batches, self.paths, self.ids, self.seen, batch_size=2)

    def rows(self, columns="image_id, image_path, rating, image_tags"):
        return self.connection.execute(f"SELECT {columns} FROM image_meta ORDER BY image_path").fetchall()

    def test_readable_images_are_stored_with_normalized_embeddings_and_sidecar_tags(self):
        extractor = self.extractor()
        self.run_extractor(extractor, self.batches())
        self.assertEqual(self.rows(), [
            ("id-1", "green.png", "", ""), ("id-0", "red.png", "general", "red, plain"),
            ("id-3", "sub/yellow.png", "", ""),
        ])
        self.assertEqual(self.seen.tolist(), [True, True, False, True])
        self.assertEqual((extractor.dimension, extractor.stored_count), (2, 3))
        stored = {
            image_id: np.frombuffer(meta, dtype=np.float32)
            for image_id, meta in self.rows("image_id, meta")
        }
        np.testing.assert_allclose(stored["id-0"], [1, 0])
        np.testing.assert_allclose(stored["id-3"], [np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-6)
        details = self.rows("aesthetic_quality, pony_aesthetic_quality, style_cluster, search_embedding_model, metadata_embedding_model")
        self.assertEqual(set(details), {(None, None, "", "fake:model", None)})
        self.assertFalse(self.connection.in_transaction)

    def test_embeddings_of_another_dimension_are_not_stored(self):
        extractor = self.extractor(dimension=3)
        self.run_extractor(extractor, self.batches())
        self.assertEqual(self.rows(), [])
        self.assertEqual((extractor.dimension, self.seen.sum()), (3, 3))

    def test_metadata_models_reuse_search_features_when_no_separate_model_is_given(self):
        aesthetic, pony, style = Aesthetic(), Pony(), Style()
        search_model = SearchModel()
        extractor = self.extractor(
            search_model=search_model, metadata_models=MetadataModels(aesthetic, pony, style),
            metadata_model_label="open_clip:ViT-L-14:openai",
        )
        self.run_extractor(extractor, self.batches(batch_size=4))
        # One forward pass per batch provides both the search and the metadata features.
        self.assertEqual(search_model.calls, 1)
        self.assertEqual((tuple(aesthetic.seen.shape), tuple(pony.seen.shape), tuple(style.seen.shape)), ((3, 2), (3, 2), (3, 4)))
        details = self.rows("image_id, aesthetic_quality, pony_aesthetic_quality, style_cluster, metadata_embedding_model")
        self.assertEqual(details[1], ("id-0", 1.0, 0.0, "style_cluster_7", "open_clip:ViT-L-14:openai"))
        self.assertAlmostEqual(details[2][2], 2 * np.sqrt(0.5), places=6)

    def test_separate_metadata_model_receives_its_own_preprocessed_input(self):
        aesthetic, pony, style = Aesthetic(), Pony(), Style()
        metadata_embedding = types.SimpleNamespace(encode_image_with_internal=mock.Mock(
            side_effect=lambda batch: (torch.ones(len(batch), 1024), torch.ones(len(batch), 768)),
        ))
        search_model = SearchModel()
        extractor = self.extractor(
            search_model=search_model, metadata_models=MetadataModels(aesthetic, pony, style),
            metadata_embedding=metadata_embedding,
        )
        self.run_extractor(extractor, self.batches(metadata_transform=lambda image: to_tensor(image) * 0, batch_size=4))
        metadata_input = metadata_embedding.encode_image_with_internal.call_args.args[0]
        self.assertEqual((tuple(metadata_input.shape), float(metadata_input.sum())), ((3, 3, 20, 20), 0.0))
        self.assertEqual((tuple(aesthetic.seen.shape), tuple(pony.seen.shape), tuple(style.seen.shape)), ((3, 768), (3, 768), (3, 1024)))
        np.testing.assert_allclose(aesthetic.seen.norm(dim=1).numpy(), 1, rtol=1e-6)
        self.assertEqual(len(self.rows()), 3)

    def test_tagging_service_receives_both_preprocessed_inputs(self):
        service = mock.Mock()
        service.tag_batch.side_effect = lambda wd, z3d: [{"rating": "explicit", "tags": ["a", "b"]}] * len(wd)
        self.run_extractor(self.extractor(tagging_service=service), self.batches(tag_sizes=(8, 16), batch_size=4))
        wd_input, z3d_input = service.tag_batch.call_args.args
        self.assertEqual((wd_input.shape, z3d_input.shape), ((3, 8, 8, 3), (3, 16, 16, 3)))
        self.assertEqual({row[2:] for row in self.rows()}, {("explicit", "a, b")})

    def test_a_failing_batch_is_skipped_and_later_batches_are_stored(self):
        search_model = SearchModel()
        encode = search_model.encode_image_with_internal
        failures = iter([RuntimeError("out of memory")])

        def fail_once(batch):
            error = next(failures, None)
            if error is not None:
                raise error
            return encode(batch)

        search_model.encode_image_with_internal = fail_once
        self.run_extractor(self.extractor(search_model=search_model), self.batches())
        self.assertEqual([row[1] for row in self.rows()], ["sub/yellow.png"])
        self.assertEqual(self.seen.tolist(), [True, True, False, True])

    def test_device_transfer_failure_reaches_the_caller_and_leaves_the_batch_unseen(self):
        batches = self.batches()
        with mock.patch.object(features, "_to_device", side_effect=RuntimeError("CUDA out of memory")):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                self.run_extractor(self.extractor(), batches)
        self.assertEqual(self.rows(), [])
        self.assertFalse(self.seen.any())

    def test_loader_failure_propagates_after_committing_stored_rows(self):
        def failing_loader():
            yield self.batches()[0]
            raise RuntimeError("DataLoader worker exited unexpectedly")

        with self.assertRaisesRegex(RuntimeError, "worker exited"):
            self.run_extractor(self.extractor(), failing_loader())
        self.assertEqual(len(self.rows()), 2)
        self.assertFalse(self.connection.in_transaction)
        self.assertEqual(self.seen.tolist(), [True, True, False, False])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        quiet(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.images = Path(temporary.name)
        save_image(self.images / "a.png", size=(8, 4), color=(10, 20, 30))
        Image.new("RGBA", (4, 4), (10, 20, 30, 0)).save(self.images / "transparent.png")
        (self.images / "broken.png").write_bytes(b"not an image")

    def samples(self, **options):
        options.setdefault("tag_sizes", None)
        return list(DirImageIterable(str(self.images), ["a.png", "broken.png", "transparent.png"], **options))

    def test_untransformed_images_are_rgb_and_unused_inputs_are_none(self):
        samples = self.samples()
        self.assertEqual([sample[2] for sample in samples], [0, 2])
        for image, metadata_input, _index, wd_input, z3d_input in samples:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual((metadata_input, wd_input, z3d_input), (None, None, None))
        self.assertEqual(samples[1][0].getpixel((0, 0)), (10, 20, 30))
        search, metadata, indexes, wd, z3d = safe_collate(samples)
        self.assertEqual((search, metadata, wd, z3d), ([samples[0][0], samples[1][0]], None, None, None))
        self.assertEqual(indexes.tolist(), [0, 2])

    def test_tagger_inputs_are_shared_when_both_taggers_use_the_same_size(self):
        same = self.samples(tag_sizes=(16, 16))[0]
        self.assertIs(same[3], same[4])
        self.assertEqual(same[3].shape, (16, 16, 3))
        different = self.samples(tag_sizes=(16, 8))[0]
        self.assertEqual((different[3].shape, different[4].shape), ((16, 16, 3), (8, 8, 3)))

    def test_transforms_are_applied_per_model_and_tensors_are_stacked(self):
        samples = self.samples(transform=lambda image: torch.zeros(3, 2, 2), metadata_transform=lambda image: torch.ones(1))
        search, metadata, _indexes, _wd, _z3d = safe_collate(samples)
        self.assertEqual((tuple(search.shape), metadata.tolist()), ((2, 3, 2, 2), [[1.0], [1.0]]))
        self.assertIsNone(safe_collate([]))
        self.assertIsNone(safe_collate([None]))

    def test_failing_transform_skips_only_that_image(self):
        def transform(image):
            if image.size == (8, 4):
                raise ValueError("unsupported")
            return torch.zeros(1)

        self.assertEqual([sample[2] for sample in self.samples(transform=transform)], [2])


class ScoringTests(unittest.TestCase):
    def test_style_cluster_dimension_mismatch_has_clear_error(self):
        cluster = StyleCluster.__new__(StyleCluster)
        cluster.cluster_centers = torch.zeros(4, 1024)
        cluster.kept_cluster_indices = list(range(4))
        with self.assertRaisesRegex(ValueError, r"features=768.*cluster_centers=1024"):
            cluster.get_cluster_batch(torch.zeros(2, 768))
        with self.assertRaisesRegex(ValueError, r"shape \(N, D\)"):
            cluster.get_cluster_batch(torch.zeros(1024))

    def test_style_cluster_returns_original_ids_of_the_nearest_kept_centers(self):
        cluster = StyleCluster.__new__(StyleCluster)
        cluster.cluster_centers = torch.tensor([[0., 0.], [10., 0.]])
        cluster.kept_cluster_indices = [5, 9]
        cluster_ids, distances = cluster.get_cluster_batch(np.array([[9., 0.], [1., 0.]], dtype=np.float64))
        self.assertEqual(cluster_ids, [9, 5])
        np.testing.assert_allclose(distances, [1, 1])

    def test_unreadable_weights_stop_the_run_instead_of_scoring_with_untrained_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"not a checkpoint")
            for load in (PonyAestheticScorer, StyleCluster):
                with self.subTest(load=load.__name__), self.assertRaisesRegex(RuntimeError, "Could not load"):
                    load(str(path), "cpu")

    def test_style_centers_exclude_blocklisted_clusters(self):
        quiet(self)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "centers.npz"
            centers = np.arange(60, dtype=np.float32).reshape(30, 2)
            np.savez(path, cluster_centers=centers)
            cluster = StyleCluster(str(path), "cpu")
        # 24 and 26 are blocklisted.
        self.assertEqual(cluster.kept_cluster_indices, [i for i in range(30) if i not in (24, 26)])
        self.assertEqual(tuple(cluster.cluster_centers.shape), (28, 2))


class RecalculationTests(unittest.TestCase):
    def test_scores_are_recomputed_from_stored_vectors_of_the_expected_dimension(self):
        quiet(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        connection = sqlite3.connect(Path(temporary.name) / "sqlite_image_meta.db")
        self.addCleanup(connection.close)
        init_db(connection)
        vectors = {"a": [3., 4.], "b": [0., 2.], "single": [1., 0.]}
        connection.executemany(
            "INSERT INTO image_meta (image_id, image_path, meta) VALUES (?, ?, ?)",
            [(key, f"{key}.png", np.array(value, dtype=np.float32).tobytes()) for key, value in vectors.items()]
            + [("other-dimension", "o.png", np.zeros(3, dtype=np.float32).tobytes()), ("no-meta", "n.png", None)],
        )
        connection.commit()
        # The last batch holds a single row.
        updated = recalc_aesthetic_scores(connection, Aesthetic(), Pony(), "cpu", 2, batch_size=2, num_workers=0)
        self.assertEqual(updated, 3)
        scores = {
            image_id: (aesthetic, pony) for image_id, aesthetic, pony in connection.execute(
                "SELECT image_id, aesthetic_quality, pony_aesthetic_quality FROM image_meta"
            )
        }
        np.testing.assert_allclose(scores["a"], (0.6, 1.6), rtol=1e-6)
        np.testing.assert_allclose(scores["b"], (0.0, 2.0), rtol=1e-6)
        np.testing.assert_allclose(scores["single"], (1.0, 0.0), rtol=1e-6)
        self.assertEqual((scores["other-dimension"], scores["no-meta"]), ((None, None), (None, None)))


if __name__ == "__main__":
    unittest.main()
