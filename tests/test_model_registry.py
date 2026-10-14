from pathlib import Path
import tempfile
import unittest

from app.domain.domain_object import ModelId
from app.domain.errors import SearchInputError
from app.infrastructure.index_files import IndexFiles, SearchModelMetadata, write_model_metadata
from app.infrastructure.model_registry import ModelRegistry


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def create_index_dir(self, parent: Path, name: str) -> Path:
        directory = parent / name
        directory.mkdir(parents=True)
        (directory / "metafiles.index").write_bytes(b"index")
        (directory / "sqlite_image_meta.db").write_bytes(b"sqlite")
        return directory

    def test_only_complete_index_directories_are_registered(self):
        self.create_index_dir(self.root, "ViT-L-14-openai")
        incomplete = self.root / "ViT-B-32-openai"
        incomplete.mkdir()
        (incomplete / "metafiles.index").write_bytes(b"index")
        (self.root / "notes.txt").write_text("not an index")

        registry = ModelRegistry(self.root)

        self.assertEqual(
            [(item.id, item.display_name) for item in registry.items()],
            [(ModelId("ViT-L-14", "openai"), "ViT-L-14-openai")],
        )
        self.assertEqual(
            registry.files(ModelId("ViT-L-14", "openai")),
            IndexFiles((self.root / "ViT-L-14-openai").resolve()),
        )

    def test_metadata_restores_model_ids_that_are_unsafe_as_directory_names(self):
        for dirname in ("Qwen--Qwen3-VL-Embedding-2B", "custom-index"):
            with self.subTest(dirname=dirname), tempfile.TemporaryDirectory(dir=self.root) as case:
                directory = self.create_index_dir(Path(case), dirname)
                write_model_metadata(directory, SearchModelMetadata(
                    "Qwen/Qwen3-VL-Embedding-2B", "", "Qwen embedding",
                ))
                registry = ModelRegistry(Path(case))
                model_id = ModelId("Qwen/Qwen3-VL-Embedding-2B", "")
                self.assertEqual([(item.id, item.display_name) for item in registry.items()], [(model_id, "Qwen embedding")])
                self.assertEqual(registry.files(model_id).directory, directory.resolve())

    def test_directory_without_a_pretrained_suffix_or_valid_metadata_uses_its_name(self):
        self.create_index_dir(self.root, "standalone")
        broken = self.create_index_dir(self.root, "Other-weights")
        (broken / "model_meta.json").write_text("[not an object]", encoding="utf-8")
        # Saved by an editor in another encoding than UTF-8.
        legacy = self.create_index_dir(self.root, "Legacy-encoding")
        (legacy / "model_meta.json").write_text('{"model_name": "x", "display_name": "日本語"}', encoding="cp932")
        # Valid JSON that does not describe a model.
        for name, content in (("List-shaped", '["not an object"]'), ("Numeric-name", '{"model_name": 5}')):
            (self.create_index_dir(self.root, name) / "model_meta.json").write_text(content, encoding="utf-8")
        with self.assertLogs("app.infrastructure.index_files", level="WARNING") as logs:
            registry = ModelRegistry(self.root)
        self.assertEqual(
            [(item.id, item.display_name) for item in registry.items()],
            [
                (ModelId("Legacy", "encoding"), "Legacy-encoding"),
                (ModelId("List", "shaped"), "List-shaped"),
                (ModelId("Numeric", "name"), "Numeric-name"),
                (ModelId("Other", "weights"), "Other-weights"),
                (ModelId("standalone", ""), "standalone"),
            ],
        )
        # Every unusable file is named once; the directory without one is not mentioned.
        self.assertEqual(len(logs.output), 4)
        for name in ("Legacy-encoding", "List-shaped", "Numeric-name", "Other-weights"):
            self.assertEqual(sum(name in line and "model_meta.json" in line for line in logs.output), 1, name)

    def test_models_are_listed_by_display_name(self):
        for name, display in (("b-x", "zeta"), ("a-x", "Beta"), ("c-x", "alpha")):
            write_model_metadata(self.create_index_dir(self.root, name), SearchModelMetadata(name, "", display))
        self.assertEqual([item.display_name for item in ModelRegistry(self.root).items()], ["alpha", "Beta", "zeta"])

    def test_unregistered_and_external_models_are_rejected(self):
        self.create_index_dir(self.root, "Test-model")
        with tempfile.TemporaryDirectory() as external:
            self.create_index_dir(Path(external), "outside-model")
            registry = ModelRegistry(self.root)
            for model_id in (
                ModelId("unknown", ""),
                ModelId(f"../{Path(external).name}/outside", "model"),
                ModelId(str(Path(external) / "outside"), "model"),
            ):
                with self.subTest(model_id=model_id):
                    with self.assertRaisesRegex(SearchInputError, "not registered") as raised:
                        registry.files(model_id)
                    # The message is returned to API clients.
                    self.assertNotIn(str(self.root), str(raised.exception))
                    self.assertNotIn(self.root.name, str(raised.exception))

    def test_missing_metadata_root_has_no_models(self):
        self.assertEqual(ModelRegistry(self.root / "missing").items(), [])


if __name__ == "__main__":
    unittest.main()
