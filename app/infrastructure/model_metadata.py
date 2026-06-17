"""Search-index model metadata helpers."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

from app.domain.domain_object import ModelId, ModelItem, ModelName

MODEL_METADATA_FILENAME = "model_meta.json"
INDEX_FILENAME = "metafiles.index"
SQLITE_METADATA_FILENAME = "sqlite_image_meta.db"


@dataclass(frozen=True)
class SearchModelMetadata:
    """Metadata that restores a ModelItem from an index directory name."""

    model_name: str
    pretrained: str
    display_name: str

    def to_model_item(self) -> ModelItem:
        return ModelItem(
            ModelId(self.model_name, self.pretrained),
            ModelName(self.display_name),
        )


def safe_model_dir_name(value: str) -> str:
    """Convert a repository/model id into a path-safe directory name."""
    return value.replace("/", "--").replace("\\", "--").replace(":", "_")


def default_model_dir_name(model_id: ModelId) -> str:
    return f"{model_id.model_name}-{model_id.pretrained}"


def has_search_index(directory: pathlib.Path) -> bool:
    return (
        directory.is_dir()
        and (directory / INDEX_FILENAME).is_file()
        and (directory / SQLITE_METADATA_FILENAME).is_file()
    )


def read_model_metadata(directory: pathlib.Path) -> SearchModelMetadata | None:
    path = directory / MODEL_METADATA_FILENAME
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    model_name = data.get("model_name")
    pretrained = data.get("pretrained", "")
    display_name = data.get("display_name")
    if not isinstance(model_name, str) or not isinstance(pretrained, str):
        return None
    if not isinstance(display_name, str) or not display_name:
        display_name = f"{model_name}-{pretrained}" if pretrained else model_name
    return SearchModelMetadata(model_name, pretrained, display_name)


def write_model_metadata(directory: pathlib.Path, metadata: SearchModelMetadata) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MODEL_METADATA_FILENAME).write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
