"""1つの検索モデルのインデックスを構成するファイルと、その読み書き。

インデックスを作成する側(indexer)と検索する側(app)が、同じファイル名・形式を
この定義から参照する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
import pathlib
import sqlite3
from urllib.parse import quote

from app.domain.domain_object import ModelId, ModelItem

logger = logging.getLogger(__name__)

INDEX_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class IndexFiles:
    """メタデータルート直下の、1モデル分のディレクトリに置かれるファイル。"""

    directory: pathlib.Path

    @property
    def database(self) -> pathlib.Path:
        """画像ごとのメタデータと埋め込みを保持するSQLite。"""
        return self.directory / "sqlite_image_meta.db"

    @property
    def index(self) -> pathlib.Path:
        return self.directory / "metafiles.index"

    @property
    def mean(self) -> pathlib.Path:
        """indexの中心化に使った平均ベクトル。"""
        return self.index.with_name(self.index.name + ".mean.npy")

    @property
    def sidecar(self) -> pathlib.Path:
        """indexの作成条件。存在すれば中心化済みの形式である。"""
        return self.index.with_name(self.index.name + ".meta.json")

    @property
    def manifest(self) -> pathlib.Path:
        """indexの行番号と画像IDの対応、およびindex・平均ベクトルのチェックサム。"""
        return self.index.with_name(self.index.name + ".rows.sqlite3")

    @property
    def model_metadata(self) -> pathlib.Path:
        return self.directory / "model_meta.json"

    def all(self) -> tuple[pathlib.Path, ...]:
        return (self.database, self.index, self.mean, self.sidecar, self.manifest, self.model_metadata)

    def has_search_index(self) -> bool:
        return self.directory.is_dir() and self.index.is_file() and self.database.is_file()


def readonly_database_uri(path: pathlib.Path) -> str:
    """読み取り専用で開くためのSQLite URI。

    ``Path.as_uri()`` はUNCパスのサーバー名をURIのauthorityに置くが、SQLiteはそれを受け付けない。
    authorityを常に空にして、ネットワーク上のメタデータも開けるようにする。
    """
    location = quote(os.fsencode(path.resolve().as_posix()))
    return "file://" + ("" if location.startswith("/") else "/") + location + "?mode=ro"


def open_readonly_database(path: pathlib.Path) -> sqlite3.Connection:
    """Never create or modify a source metadata/manifest database while searching."""
    return sqlite3.connect(readonly_database_uri(path), uri=True)


def file_sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# =====================================================
# 画像メタデータ(sqlite_image_meta.db の image_meta テーブル)

def image_meta_columns(connection: sqlite3.Connection) -> frozenset[str]:
    columns = frozenset(row[1] for row in connection.execute("PRAGMA table_info(image_meta)"))
    if not {"image_id", "image_path"} <= columns:
        raise ValueError("Image metadata must contain image_id and image_path columns.")
    return columns


def select_list(available: frozenset[str], names: tuple[str, ...], prefix: str = "") -> str:
    """Older databases predate optional rating/style/aesthetic columns; read those as NULL."""
    return ", ".join(f"{prefix}{name}" if name in available else f"NULL AS {name}" for name in names)


# =====================================================
# モデルの識別情報(model_meta.json)

@dataclass(frozen=True)
class SearchModelMetadata:
    """Metadata that restores a ModelItem from an index directory name."""

    model_name: str
    pretrained: str
    display_name: str

    def to_model_item(self) -> ModelItem:
        return ModelItem(ModelId(self.model_name, self.pretrained), self.display_name)


def safe_model_dir_name(value: str) -> str:
    """Convert a repository/model id into a path-safe directory name."""
    return value.replace("/", "--").replace("\\", "--").replace(":", "_")


def read_model_metadata(directory: pathlib.Path) -> SearchModelMetadata | None:
    path = IndexFiles(directory).model_metadata
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("model metadata must be an object")
        model_name = data.get("model_name")
        pretrained = data.get("pretrained", "")
        if not isinstance(model_name, str) or not isinstance(pretrained, str):
            raise ValueError("model_name and pretrained must be strings")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        # UTF-8以外で保存されたファイルや、形式の違うJSONも、壊れたJSONと同じく無視する。
        logger.warning("モデル情報を読めないため、ディレクトリ名から判断します: %s (%s)", path, error)
        return None

    display_name = data.get("display_name")
    if not isinstance(display_name, str) or not display_name:
        display_name = f"{model_name}-{pretrained}" if pretrained else model_name
    return SearchModelMetadata(model_name, pretrained, display_name)


def write_model_metadata(directory: pathlib.Path, metadata: SearchModelMetadata) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    IndexFiles(directory).model_metadata.write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def model_item_from_directory(directory: pathlib.Path) -> ModelItem:
    """model_meta.jsonがなければ、``<model_name>-<pretrained>`` 形式のディレクトリ名から復元する。"""
    metadata = read_model_metadata(directory)
    if metadata is not None:
        return metadata.to_model_item()
    model_name, separator, pretrained = directory.name.rpartition("-")
    if not separator:
        model_name, pretrained = directory.name, ""
    return ModelItem(ModelId(model_name, pretrained), directory.name)


# =====================================================
# indexの作成条件(.meta.json)

def write_index_sidecar(path: pathlib.Path, *, embedding_dim: int, pq_m: int) -> None:
    path.write_text(json.dumps({
        "embedding_dim": embedding_dim, "pq_m": pq_m, "mean_centered": True,
        "rows_manifest_version": INDEX_MANIFEST_VERSION,
    }, indent=2) + "\n", encoding="utf-8")


def sidecar_requires_manifest(path: pathlib.Path) -> bool:
    """行マニフェストを伴う形式で作成されたindexかどうか。sidecarがなければFalse。"""
    if not path.is_file():
        return False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("index metadata must be an object")
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid index metadata: {path}") from error
    return metadata.get("rows_manifest_version") is not None


# =====================================================
# 行マニフェスト(.rows.sqlite3)

@dataclass(frozen=True)
class IndexManifest:
    """マニフェストのヘッダー。行の一覧は保持しない。"""

    count: int
    dimension: int
    index_sha256: str
    mean_sha256: str


def create_manifest_tables(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE index_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("""CREATE TABLE index_rows (
        position INTEGER PRIMARY KEY, image_id TEXT NOT NULL UNIQUE, image_path TEXT NOT NULL
    )""")


def write_manifest_header(connection: sqlite3.Connection, manifest: IndexManifest) -> None:
    header = {"version": INDEX_MANIFEST_VERSION, "mean_centered": True, **asdict(manifest)}
    connection.executemany(
        "INSERT INTO index_info VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in header.items()],
    )


def read_index_manifest(
    manifest_path: pathlib.Path, *, connection: sqlite3.Connection | None = None, schema: str = "main"
) -> IndexManifest | None:
    """Read and validate the manifest header. Returns None when no manifest exists.

    Pass ``connection`` and ``schema`` to read a manifest attached to another
    database, so the header and the rows come from one SQLite snapshot.
    """
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
        return IndexManifest(
            header["count"], header["dimension"], header["index_sha256"], header["mean_sha256"]
        )
    except (sqlite3.Error, OSError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid index manifest {manifest_path}: {error}. Rebuild the index.") from error
