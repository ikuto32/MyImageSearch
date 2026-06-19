import pathlib
import tempfile
from unittest import TestCase

from app.domain.domain_object import ModelId, ModelName
from app.infrastructure.local_repository import LocalRepository
from app.infrastructure.model_metadata import SearchModelMetadata, write_model_metadata


class LocalRepositoryModelListTest(TestCase):
    def _repository(self, meta_dir: pathlib.Path) -> LocalRepository:
        return LocalRepository(pathlib.Path("/unused/images"), meta_dir)

    def _create_index_dir(self, meta_dir: pathlib.Path, dirname: str) -> pathlib.Path:
        index_dir = meta_dir / dirname
        index_dir.mkdir(parents=True)
        (index_dir / "metafiles.index").write_bytes(b"index")
        (index_dir / "sqlite_image_meta.db").write_bytes(b"sqlite")
        return index_dir

    def test_load_all_model_item_uses_only_clip_meta_index_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            meta_dir = pathlib.Path(temp_dir_name)
            self._create_index_dir(meta_dir, "ViT-L-14-openai")
            incomplete_dir = meta_dir / "ViT-B-32-openai"
            incomplete_dir.mkdir()
            (incomplete_dir / "metafiles.index").write_bytes(b"index")

            repo = self._repository(meta_dir)

            items = repo.load_all_model_item()

            self.assertEqual(
                [item.id for item in items], [ModelId("ViT-L-14", "openai")]
            )
            self.assertEqual(
                [item.display_name for item in items], [ModelName("ViT-L-14-openai")]
            )

    def test_load_all_model_item_restores_metadata_for_safe_directory_names(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            meta_dir = pathlib.Path(temp_dir_name)
            index_dir = self._create_index_dir(meta_dir, "Qwen--Qwen3-VL-Embedding-2B")
            write_model_metadata(
                index_dir,
                SearchModelMetadata(
                    "Qwen/Qwen3-VL-Embedding-2B",
                    "",
                    "Qwen/Qwen3-VL-Embedding-2B",
                ),
            )

            items = self._repository(meta_dir).load_all_model_item()

            self.assertEqual(
                [item.id for item in items],
                [ModelId("Qwen/Qwen3-VL-Embedding-2B", "")],
            )
            self.assertEqual(
                [item.display_name for item in items],
                [ModelName("Qwen/Qwen3-VL-Embedding-2B")],
            )
