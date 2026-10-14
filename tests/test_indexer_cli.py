from contextlib import closing, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import faiss
import numpy as np
from PIL import Image
import torch

from app.application.usecase import Usecase
from app.domain.domain_object import ModelId
from app.domain.search_query import SearchQuery
from app.infrastructure.index_files import IndexFiles, file_sha256, read_model_metadata
from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.local_repository import LocalRepository
from app.infrastructure.model_registry import ModelRegistry
from app.presentation.controller import create_app
from indexer import cli, faiss_index
from helpers import save_image

COLORS = {"red.png": (255, 0, 0), "green.png": (0, 255, 0), os.path.join("sub", "blue.png"): (0, 0, 255)}


def to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255)


class ColorSearchModel:
    """Embeds an image as its mean colour, so similar colours are near each other."""

    preprocess = staticmethod(to_tensor)

    def encode_image_with_internal(self, batch):
        features = batch.mean(dim=(2, 3))
        return features, features


def quiet(test: unittest.TestCase) -> None:
    for redirect in (redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
        redirect.__enter__()
        test.addCleanup(redirect.__exit__, None, None, None)


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict(os.environ, {
            "MYIMAGESEARCH_IMAGE_DIR": "", "MYIMAGESEARCH_META_DIR": "", "VLLM_API_BASE": "http://vllm:9000", "VLLM_API_KEY": "",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_defaults_use_project_data_and_the_qwen_api_backend(self):
        args = cli.parse_arguments([])
        self.assertEqual(Path(args.image_dir), cli.PROJECT_DIR / "images")
        self.assertEqual(Path(args.meta_dir), cli.PROJECT_DIR / "clip_meta")
        self.assertEqual((args.search_backend, args.search_model_id), ("qwen_vl", "Qwen/Qwen3-VL-Embedding-2B"))
        self.assertIsNone(args.search_model_out_dim)
        self.assertEqual((args.qwen_api_base, args.qwen_api_timeout, args.qwen_api_max_retries, args.qwen_api_concurrency),
                         ("http://vllm:9000", 300.0, 2, 4))
        self.assertEqual((args.batch_size, args.num_workers, args.nlist, args.pq_m, args.bits_per_code), (32, 8, 64, 64, 8))
        self.assertFalse(args.use_existing_tags or args.disable_clip_metadata or args.recalc_aesthetic)

    def test_environment_and_explicit_paths_select_the_data_location(self):
        os.environ["MYIMAGESEARCH_IMAGE_DIR"] = "external-images"
        self.assertEqual(cli.parse_arguments([]).image_dir, "external-images")
        self.assertEqual(cli.parse_arguments(["--image-dir", "explicit"]).image_dir, "explicit")

    def test_invalid_numbers_are_rejected(self):
        quiet(self)
        for arguments in (["--batch-size", "0"], ["--num-workers", "-1"], ["--search-model-out-dim", "0"], ["--qwen-api-timeout", "0"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                cli.parse_arguments(arguments)

    def test_aesthetic_recalculation_is_refused_for_embeddings_of_other_models(self):
        quiet(self)
        for arguments in ([], ["--search-backend", "open_clip", "--search-model-pretrained", "laion2b"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                cli.parse_arguments(["--recalc-aesthetic", *arguments])
        self.assertTrue(cli.parse_arguments(["--recalc-aesthetic", "--search-backend", "open_clip"]).recalc_aesthetic)

    def test_index_directory_and_model_identity_follow_the_search_backend(self):
        with tempfile.TemporaryDirectory() as meta:
            for arguments, directory, identity in (
                (["--search-backend", "open_clip"], "ViT-L-14-openai", ("ViT-L-14", "openai", "ViT-L-14-openai")),
                (["--search-model-id", "Qwen/Qwen3:8B"], "Qwen--Qwen3_8B", ("Qwen/Qwen3:8B", "", "Qwen/Qwen3:8B")),
            ):
                with self.subTest(directory=directory):
                    args = cli.parse_arguments(["--meta-dir", meta, *arguments])
                    files = cli.prepare_index_files(args)
                    self.assertEqual(files, IndexFiles(Path(meta) / directory))
                    metadata = read_model_metadata(files.directory)
                    self.assertEqual((metadata.model_name, metadata.pretrained, metadata.display_name), identity)

    def test_labels_and_shared_metadata_model(self):
        qwen = cli.parse_arguments([])
        self.assertEqual(cli.search_embedding_model_label(qwen), "qwen_vl:Qwen/Qwen3-VL-Embedding-2B")
        self.assertEqual(cli.metadata_embedding_model_label(qwen), "open_clip:ViT-L-14:openai")
        self.assertFalse(cli.uses_metadata_model_for_search(qwen))
        clip = cli.parse_arguments(["--search-backend", "open_clip", "--disable-clip-metadata"])
        self.assertEqual(cli.search_embedding_model_label(clip), "open_clip:ViT-L-14:openai")
        self.assertIsNone(cli.metadata_embedding_model_label(clip))
        self.assertTrue(cli.uses_metadata_model_for_search(clip))
        other = cli.parse_arguments(["--search-backend", "open_clip", "--search-model-pretrained", "laion2b"])
        self.assertFalse(cli.uses_metadata_model_for_search(other))


class IndexingWorkflowTests(unittest.TestCase):
    """The command line builds an index from images that the web application can search."""

    def setUp(self):
        quiet(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.images = Path(temporary.name) / "images"
        self.meta = Path(temporary.name) / "meta"
        for name, color in COLORS.items():
            save_image(self.images / name, color=color)
        (self.images / "red.png.tags.json").write_text(
            json.dumps({"rating": "general", "tags": ["plain", "warm"]}), encoding="utf-8",
        )
        (self.images / "notes.txt").write_text("not an image")
        self.model = ModelId("Fake", "test")
        self.files = IndexFiles(self.meta / "Fake-test")
        self.load_backend = mock.Mock(return_value=ColorSearchModel())

    def run_indexer(self, *extra):
        arguments = [
            "--image-dir", str(self.images), "--meta-dir", str(self.meta),
            "--search-backend", "open_clip", "--search-model-name", "Fake", "--search-model-pretrained", "test",
            "--use-existing-tags", "--disable-clip-metadata", "--num-workers", "0", "--batch-size", "2", *extra,
        ]
        with (
            mock.patch.object(cli, "load_search_embedding_backend", self.load_backend),
            mock.patch.object(faiss_index, "create_ivfpq_index", side_effect=lambda *_: faiss.IndexFlatIP(3)),
        ):
            cli.main(arguments)

    def client(self):
        registry = ModelRegistry(self.meta)
        usecase = Usecase(LocalRepository(self.images, registry), LocalAccessor(registry))
        return create_app(usecase).test_client()

    def search(self, client, route, **fields):
        params = {"model_name": "Fake", "pretrained": "test", **fields}
        response = client.post(route, json={"params": params})
        self.assertEqual(response.status_code, 200, response.data)
        return [(result["item"]["name"], result["item"]["rating"]) for result in response.get_json()["list"]]

    def test_images_are_indexed_once_and_searchable_by_the_application(self):
        self.run_indexer()
        self.load_backend.assert_called_once()
        self.assertTrue(all(path.is_file() for path in self.files.all()))

        client = self.client()
        self.assertEqual(client.get("/model_item").get_json(), [{"model_name": "Fake", "pretrained": "test"}])
        page = client.get("/image_item", query_string={"model_name": "Fake", "pretrained": "test", "include_total": 1})
        self.assertEqual([item["name"] for item in page.get_json()], sorted(COLORS))
        self.assertEqual(page.headers["X-Catalog-Total"], "3")

        reddish = SearchQuery((0.9, 0.1, 0.0), True, self.model).to_text()
        self.assertEqual(self.search(client, "/search/query", search_query=reddish)[0], ("red.png", "general"))
        bluish = SearchQuery((0.0, 0.2, 0.8), True, self.model).to_text()
        self.assertEqual(self.search(client, "/search/query", search_query=bluish, result_size=1), [(os.path.join("sub", "blue.png"), "")])
        self.assertEqual(self.search(client, "/search/name", text="blue"), [(os.path.join("sub", "blue.png"), "")])
        self.assertEqual(self.search(client, "/search/tags", text="warm"), [("red.png", "general")])
        self.assertEqual(len(self.search(client, "/search/random")), 3)

        # A search result carries the ID that the catalog and the image routes use for the same file.
        found = client.post("/search/name", json={"params": {"model_name": "Fake", "pretrained": "test", "text": "red"}})
        image_id = found.get_json()["list"][0]["item"]["id"]
        self.assertEqual(image_id, hashlib.sha256(b"red.png").hexdigest())
        self.assertEqual(image_id, {item["name"]: item["id"] for item in page.get_json()}["red.png"])
        thumbnail = client.get(f"/image/{image_id}/small")
        self.assertEqual((thumbnail.status_code, thumbnail.content_type), (200, "image/png"))
        with Image.open(io.BytesIO(thumbnail.data)) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

        # A second run finds nothing to do: no model is loaded and the same index is produced.
        index_hash = file_sha256(self.files.index)
        self.run_indexer()
        self.load_backend.assert_called_once()
        self.assertEqual(file_sha256(self.files.index), index_hash)

    def test_later_runs_add_new_images_and_remove_deleted_ones_on_request(self):
        self.run_indexer()
        save_image(self.images / "yellow.png", color=(255, 255, 0))
        (self.images / "green.png").unlink()

        self.run_indexer()
        client = self.client()
        names = [name for name, _ in self.search(client, "/search/name", text=".png")]
        self.assertEqual(names, sorted([*COLORS, "yellow.png"]))

        self.run_indexer("--delete-orphan-db-records")
        names = [name for name, _ in self.search(self.client(), "/search/name", text=".png")]
        self.assertEqual(names, sorted([name for name in COLORS if name != "green.png"] + ["yellow.png"]))

    def stored_paths(self):
        with closing(sqlite3.connect(self.files.database)) as connection:
            return sorted(row[0] for row in connection.execute("SELECT image_path FROM image_meta"))

    def test_orphans_are_kept_when_the_image_directory_cannot_be_read(self):
        self.run_indexer()
        index_hash = file_sha256(self.files.index)
        # An unmounted share or a mistyped path looks like a directory without images.
        self.run_indexer("--image-dir", str(self.images / "unreachable"), "--delete-orphan-db-records")
        self.assertEqual(self.stored_paths(), sorted(COLORS))
        self.assertEqual(file_sha256(self.files.index), index_hash)

    def delete_orphans_after_partial_scan(self, unreadable, hidden):
        """Delete orphans while the scan does not list paths starting with ``hidden`` and reports ``unreadable``."""
        complete_scan = cli.scan_image_dir

        def partial_scan(*args, **options):
            scan = complete_scan(*args, **options)
            return scan._replace(
                paths=[path for path in scan.paths if not path.startswith(hidden)], unreadable=list(unreadable),
            )

        with mock.patch.object(cli, "scan_image_dir", side_effect=partial_scan):
            self.run_indexer("--delete-orphan-db-records")

    def test_only_orphans_below_an_unreadable_directory_are_kept(self):
        save_image(self.images / "sub2" / "cyan.png", color=(0, 255, 255))
        save_image(self.images / "subway.png", color=(9, 9, 9))
        self.run_indexer()
        for name in ("green.png", os.path.join("sub2", "cyan.png"), "subway.png"):
            (self.images / name).unlink()
        # Names that merely start like the directory are not below it. Windows names ignore case.
        self.delete_orphans_after_partial_scan(["SUB" if os.name == "nt" else "sub"], hidden="sub" + os.sep)
        self.assertEqual(self.stored_paths(), sorted(name for name in COLORS if name != "green.png"))

    def test_every_orphan_is_kept_when_the_image_directory_itself_was_not_listed_completely(self):
        self.run_indexer()
        (self.images / "green.png").unlink()
        self.delete_orphans_after_partial_scan([""], hidden="sub" + os.sep)
        self.assertEqual(self.stored_paths(), sorted(COLORS))

    def test_embeddings_of_several_dimensions_stop_the_run_before_the_image_directory_is_scanned(self):
        self.run_indexer()
        with closing(sqlite3.connect(self.files.database)) as connection:
            connection.execute(
                "INSERT INTO image_meta (image_id, image_path, meta) VALUES ('gone', 'gone.png', ?)",
                (np.zeros(5, dtype=np.float32).tobytes(),),
            )
            connection.commit()
        with mock.patch.object(cli, "scan_image_dir") as scan:
            with self.assertRaisesRegex(ValueError, "different dimensions.*--delete-orphan-db-records"):
                self.run_indexer()
        scan.assert_not_called()
        # Choosing the dimension and removing the leftover record in the same run ends that state.
        self.run_indexer("--search-model-out-dim", "3", "--delete-orphan-db-records")
        self.run_indexer()
        self.assertEqual(self.stored_paths(), sorted(COLORS))

    def test_file_names_the_console_cannot_encode_do_not_stop_the_run(self):
        self.run_indexer()
        (self.images / "broken \u2665.png").write_bytes(b"not an image")
        output = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
        with redirect_stdout(output):
            self.run_indexer()
        output.flush()
        self.assertIn(rb"broken \u2665.png", output.buffer.getvalue())

    def test_files_that_cannot_be_opened_do_not_load_the_model_on_every_run(self):
        self.run_indexer()
        index_hash = file_sha256(self.files.index)
        (self.images / "broken.png").write_bytes(b"not an image")
        self.run_indexer()
        self.load_backend.assert_called_once()
        self.assertEqual(self.stored_paths(), sorted(COLORS))
        self.assertEqual(file_sha256(self.files.index), index_hash)
        # A readable new image next to the broken file is still processed.
        save_image(self.images / "yellow.png", color=(255, 255, 0))
        self.run_indexer()
        self.assertEqual(self.load_backend.call_count, 2)
        self.assertEqual(self.stored_paths(), sorted([*COLORS, "yellow.png"]))

    def test_later_runs_keep_the_stored_embedding_dimension(self):
        self.run_indexer()
        self.assertIsNone(self.load_backend.call_args.args[2])
        save_image(self.images / "yellow.png", color=(255, 255, 0))
        self.run_indexer()
        self.assertEqual(self.load_backend.call_args.args[2], 3)

    def test_orphans_are_kept_when_the_run_fails_before_processing_images(self):
        self.run_indexer()
        save_image(self.images / "yellow.png", color=(255, 255, 0))
        (self.images / "green.png").unlink()
        self.load_backend.side_effect = RuntimeError("Could not connect to the Qwen vLLM server")
        with self.assertRaisesRegex(RuntimeError, "Could not connect"):
            self.run_indexer("--delete-orphan-db-records")
        # The published index still refers to green.png, so its record must survive.
        self.assertEqual(self.stored_paths(), sorted(COLORS))
        self.assertEqual(len(self.search(self.client(), "/search/random")), 3)

    def test_embeddings_of_another_dimension_than_requested_are_not_indexed(self):
        self.run_indexer("--search-model-out-dim", "5")
        self.assertFalse(self.files.index.exists())
        self.assertEqual(self.load_backend.call_args.args[2], 5)

    def test_empty_image_directory_reports_that_nothing_can_be_indexed(self):
        for path in self.images.rglob("*.png"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "Embedding dimension is unknown"):
            self.run_indexer()
        self.load_backend.assert_not_called()


class RecordingClipBackend(ColorSearchModel):
    """Stands in for OpenCLIP: remembers which models were built and how many batches each one embedded."""

    created = []

    def __init__(self, model_name, pretrained, device):
        self.name, self.batches = f"{model_name}/{pretrained}", 0
        RecordingClipBackend.created.append(self)

    def encode_image_with_internal(self, batch):
        self.batches += 1
        return super().encode_image_with_internal(batch)


class FakeMetadataModels:
    def score(self, internal, projected):
        count = len(projected)
        return np.full(count, 0.5), np.full(count, 6.0), [3] * count


class FakeTaggingService:
    input_sizes = (4, 4)

    def tag_batch(self, wd_batch, z3d_batch):
        return [{"rating": "sensitive", "tags": ["tagged"]} for _ in wd_batch]


class ModelWiringTests(unittest.TestCase):
    """Which models the command line loads, and what it passes to them."""

    def setUp(self):
        quiet(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.images = Path(temporary.name) / "images"
        self.meta = Path(temporary.name) / "meta"
        for name, color in COLORS.items():
            save_image(self.images / name, color=color)
        RecordingClipBackend.created = []
        self.tagging = mock.Mock(side_effect=FakeTaggingService)

    def run_indexer(self, *extra):
        with (
            mock.patch.object(cli, "OpenClipEmbeddingBackend", RecordingClipBackend),
            mock.patch.object(cli, "load_metadata_models", return_value=FakeMetadataModels()),
            mock.patch.object(cli, "ImageTaggingService", self.tagging),
            mock.patch.object(faiss_index, "create_ivfpq_index", side_effect=lambda *_: faiss.IndexFlatIP(3)),
        ):
            cli.main([
                "--image-dir", str(self.images), "--meta-dir", str(self.meta), "--search-backend", "open_clip",
                "--num-workers", "0", "--batch-size", "2", *extra,
            ])

    def models(self):
        return [(backend.name, backend.batches) for backend in RecordingClipBackend.created]

    def stored(self, directory):
        with closing(sqlite3.connect(IndexFiles(self.meta / directory).database)) as connection:
            return set(connection.execute(
                "SELECT aesthetic_quality, pony_aesthetic_quality, style_cluster, rating, image_tags,"
                " search_embedding_model, metadata_embedding_model FROM image_meta"
            ))

    def test_the_metadata_model_embeds_each_batch_once_when_it_is_also_the_search_model(self):
        self.run_indexer()
        self.assertEqual(self.models(), [("ViT-L-14/openai", 2)])
        self.assertEqual(self.stored("ViT-L-14-openai"), {(
            0.5, 6.0, "style_cluster_3", "sensitive", "tagged", "open_clip:ViT-L-14:openai", "open_clip:ViT-L-14:openai",
        )})

    def test_another_search_model_gets_a_separate_metadata_pass(self):
        self.run_indexer("--search-model-pretrained", "laion2b")
        self.assertEqual(self.models(), [("ViT-L-14/laion2b", 2), ("ViT-L-14/openai", 2)])
        self.assertEqual(self.stored("ViT-L-14-laion2b"), {(
            0.5, 6.0, "style_cluster_3", "sensitive", "tagged", "open_clip:ViT-L-14:laion2b", "open_clip:ViT-L-14:openai",
        )})

    def test_images_are_prepared_for_the_taggers_only_when_they_run(self):
        with mock.patch.object(cli, "DirImageIterable", wraps=cli.DirImageIterable) as dataset:
            self.run_indexer()
            self.assertEqual(dataset.call_args.kwargs["tag_sizes"], (4, 4))
            self.tagging.assert_called_once_with()
            self.tagging.reset_mock()
            save_image(self.images / "yellow.png", color=(255, 255, 0))
            self.run_indexer("--use-existing-tags")
            self.assertIsNone(dataset.call_args.kwargs["tag_sizes"])
            self.tagging.assert_not_called()

    def test_metadata_can_be_left_out(self):
        self.run_indexer("--disable-clip-metadata", "--use-existing-tags")
        self.assertEqual(self.models(), [("ViT-L-14/openai", 2)])
        self.assertEqual(self.stored("ViT-L-14-openai"), {(None, None, "", "", "", "open_clip:ViT-L-14:openai", None)})

    def test_recalculation_reads_the_stored_embeddings_with_the_requested_workers(self):
        self.run_indexer()
        index_hash = file_sha256(IndexFiles(self.meta / "ViT-L-14-openai").index)
        with (
            mock.patch.object(cli, "recalc_aesthetic_scores", return_value=3) as recalculate,
            mock.patch.object(cli, "ensure_asset_path", return_value="checkpoint"),
            mock.patch.object(cli, "load_aesthetic_model") as aesthetic,
            mock.patch.object(cli, "PonyAestheticScorer") as pony,
        ):
            self.run_indexer("--recalc-aesthetic", "--num-workers", "3", "--recalc-aesthetic-batch-size", "7")
        arguments = recalculate.call_args
        self.assertEqual(arguments.args[1:], (aesthetic.return_value, pony.return_value, mock.ANY, 3))
        self.assertEqual(arguments.kwargs, {"batch_size": 7, "num_workers": 3})
        # No image was read again, and the index was rebuilt from the same embeddings.
        self.assertEqual(self.models(), [("ViT-L-14/openai", 2)])
        self.assertEqual(file_sha256(IndexFiles(self.meta / "ViT-L-14-openai").index), index_hash)

    def test_the_embedding_dimension_is_the_truncation_size_of_the_qwen_backend(self):
        args = cli.parse_arguments(["--search-model-id", "Qwen/Test", "--qwen-api-base", "http://vllm:1"])
        with mock.patch.object(cli, "QwenVlEmbeddingBackend") as backend:
            cli.load_search_embedding_backend(args, "cpu", 1024)
        self.assertEqual(backend.call_args.args[:3], ("Qwen/Test", 1024, "http://vllm:1"))


class FakeLoader:
    """Collates the dataset in the main process and fails where a test asks it to."""

    created = []
    failures = []

    def __init__(self, dataset, *, batch_size, collate_fn, num_workers, pin_memory, **options):
        self.dataset, self.batch_size, self.collate_fn = dataset, batch_size, collate_fn
        self.options = {"num_workers": num_workers, "pin_memory": pin_memory, **options}
        self.fail_after = FakeLoader.failures.pop(0) if FakeLoader.failures else None
        FakeLoader.created.append(self)

    def __iter__(self):
        samples = list(self.dataset)
        for number, start in enumerate(range(0, len(samples), self.batch_size)):
            if self.fail_after == number:
                raise RuntimeError("DataLoader worker exited unexpectedly")
            yield self.collate_fn(samples[start:start + self.batch_size])


class RetryTests(unittest.TestCase):
    def setUp(self):
        quiet(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.images = Path(temporary.name)
        self.paths = [f"{index}.png" for index in range(6)]
        for path in self.paths:
            save_image(self.images / path, size=(2, 2))
        FakeLoader.created, FakeLoader.failures = [], []
        self.processed = []

    def extract(self, failures):
        FakeLoader.failures = list(failures)
        extractor = mock.Mock()

        def run(loader, image_paths, image_ids, seen, *, batch_size):
            for batch in loader:
                indexes = [int(index) for index in batch[2]]
                seen[indexes] = True
                self.processed.extend(image_paths[index] for index in indexes)

        extractor.run.side_effect = run
        with mock.patch.object(cli, "DataLoader", FakeLoader):
            cli.extract_with_retries(
                extractor, self.paths, list(self.paths), image_dir=str(self.images), batch_size=2,
                num_workers=3, transform=to_tensor, metadata_transform=None, tag_sizes=None,
            )

    def test_only_unprocessed_images_are_retried_with_fewer_workers(self):
        self.extract([1, 1])
        self.assertEqual(self.processed, self.paths)
        self.assertEqual([loader.dataset.img_list for loader in FakeLoader.created], [self.paths, self.paths[2:], self.paths[4:]])
        self.assertEqual(
            [loader.options for loader in FakeLoader.created],
            [
                {"num_workers": 3, "pin_memory": True, "shuffle": False, "prefetch_factor": 16},
                {"num_workers": 1, "pin_memory": False, "shuffle": False},
                {"num_workers": 0, "pin_memory": False, "shuffle": False},
            ],
        )

    def test_a_batch_that_keeps_failing_is_skipped_so_the_run_terminates(self):
        self.extract([0, 0])
        self.assertEqual(self.processed, self.paths[4:])
        self.assertEqual(len(FakeLoader.created), 3)

    def test_successful_run_uses_one_loader(self):
        self.extract([])
        self.assertEqual((self.processed, len(FakeLoader.created)), (self.paths, 1))


if __name__ == "__main__":
    unittest.main()
