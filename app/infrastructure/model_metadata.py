"""Search-index model metadata helpers."""

from __future__ import annotations

import json
import hashlib
import pathlib
import sqlite3
from dataclasses import asdict, dataclass

from app.domain.domain_object import ModelId, ModelItem, ModelName

MODEL_METADATA_FILENAME = "model_meta.json"
INDEX_FILENAME = "metafiles.index"
SQLITE_METADATA_FILENAME = "sqlite_image_meta.db"
INDEX_MANIFEST_SUFFIX = ".rows.sqlite3"
INDEX_MANIFEST_VERSION = 1


def open_readonly_database(path: pathlib.Path) -> sqlite3.Connection:
    """Never create or modify a source metadata/manifest database while searching."""
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_index_manifest(
    index_path: pathlib.Path, *, connection: sqlite3.Connection | None = None, schema: str = "main"
) -> dict[str, object] | None:
    """Read and validate the small manifest header without retaining the row list."""
    manifest_path = pathlib.Path(str(index_path) + INDEX_MANIFEST_SUFFIX)
    if connection is None and not manifest_path.is_file():
        return None
    if schema not in {"main", "manifest"}:
        raise ValueError("Unsupported manifest database alias")
    try:
        con = connection if connection is not None else open_readonly_database(manifest_path)
        try:
            header = dict(con.execute(f"SELECT key, value FROM {schema}.index_info"))
            count, low, high = con.execute(
                f"SELECT COUNT(*), MIN(position), MAX(position) FROM {schema}.index_rows"
            ).fetchone()
        finally:
            if connection is None:
                con.close()
        header = {key: json.loads(value) for key, value in header.items()}
        if header.get("version") != INDEX_MANIFEST_VERSION:
            raise ValueError("unsupported manifest version")
        if type(header.get("count")) is not int or header["count"] != count:
            raise ValueError("manifest row count mismatch")
        if count and (low != 0 or high != count - 1):
            raise ValueError("manifest positions are not contiguous")
        if type(header.get("dimension")) is not int or header["dimension"] < 1:
            raise ValueError("invalid manifest dimension")
        for key in ("index_sha256", "mean_sha256"):
            if not isinstance(header.get(key), str) or len(header[key]) != 64:
                raise ValueError(f"invalid {key}")
        if header.get("mean_centered") is not True:
            raise ValueError("unsupported manifest preprocessing")
        return header
    except (sqlite3.Error, OSError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid index manifest {manifest_path}: {error}. Rebuild the index.") from error


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


def normalize_model_id(model_id: ModelId) -> ModelId:
    """Normalize UI-supplied model IDs to the index layout used by create_index.py.

    Non-OpenCLIP backends such as Qwen are stored by repository/model ID only
    (for example ``Qwen--Qwen3-VL-Embedding-2B``) and do not have an OpenCLIP
    pretrained tag. Older UI state can still submit ``pretrained="openai"``;
    clear that stale value so lookups do not target ``Qwen/...-openai``.
    """
    if "/" in model_id.model_name and model_id.pretrained == "openai":
        return ModelId(model_id.model_name, "")
    return model_id


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

    if not isinstance(data, dict):
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
