"""画像ごとのメタデータと埋め込みを保持するSQLiteデータベース(image_metaテーブル)。"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import pathlib
import sqlite3
from typing import Iterable, NamedTuple
import uuid

import numpy as np
import tqdm

from app.infrastructure.index_files import IndexFiles

FLOAT32_BYTES = np.dtype(np.float32).itemsize
_ID_CHUNK = 10_000

# image_metaの列。後から追加した列は、古いデータベースにも init_db が追加する。
_COLUMNS = {
    "image_id": "text PRIMARY KEY",
    "image_path": "text",
    "meta": "blob",
    "aesthetic_quality": "real",
    "pony_aesthetic_quality": "real",
    "style_cluster": "text",
    "rating": "text",
    "image_tags": "text",
    "time_stamp_ISO": "text",
    "search_embedding_model": "text",
    "metadata_embedding_model": "text",
    # 画像ファイルの内容のblake3ハッシュ(16進)。この列を追加する前に処理した画像はNULL。
    "image_blake3": "text",
}

UPSERT_IMAGE_META = """
    INSERT INTO image_meta (
        image_id, image_path, meta, aesthetic_quality, pony_aesthetic_quality,
        style_cluster, rating, image_tags, time_stamp_ISO,
        search_embedding_model, metadata_embedding_model, image_blake3
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(image_id) DO UPDATE SET
        meta=excluded.meta,
        aesthetic_quality=excluded.aesthetic_quality,
        pony_aesthetic_quality=excluded.pony_aesthetic_quality,
        style_cluster=excluded.style_cluster,
        rating=excluded.rating,
        image_tags=excluded.image_tags,
        time_stamp_ISO=excluded.time_stamp_ISO,
        search_embedding_model=excluded.search_embedding_model,
        metadata_embedding_model=excluded.metadata_embedding_model,
        image_blake3=excluded.image_blake3
"""


def connect_db(directory: str | pathlib.Path) -> sqlite3.Connection:
    return sqlite3.connect(IndexFiles(pathlib.Path(directory)).database, isolation_level="DEFERRED")


def init_db(connection: sqlite3.Connection) -> None:
    connection.execute("pragma journal_mode = WAL")
    connection.execute("pragma synchronous = normal")
    connection.execute("pragma temp_store = memory")
    connection.execute("pragma mmap_size = 1073741824")  # 1 GB

    definitions = ", ".join(f"{name} {kind}" for name, kind in _COLUMNS.items())
    connection.execute(f"create table if not exists image_meta({definitions})")
    # SQLiteの列名は大文字と小文字を区別しない。
    existing = {row[1].lower() for row in connection.execute("PRAGMA table_info(image_meta)")}
    for name, kind in _COLUMNS.items():
        if name.lower() not in existing:
            connection.execute(f"ALTER TABLE image_meta ADD COLUMN {name} {kind}")


def finalize_db(connection: sqlite3.Connection) -> None:
    """閲覧と集計に使うindexを用意し、WALをデータベース本体へ反映する。"""
    # image_idは主キーのindexで引けるため、同じ列の重複したindexは持たない。
    connection.execute("DROP INDEX IF EXISTS idx_image_meta_image_id")
    # 一覧の並び順(image_path)と、区分ごとの件数集計(rating)を表の走査なしで行う。
    for name, column in (("idx_image_meta_path", "image_path"), ("idx_image_meta_rating", "rating")):
        statement = f"CREATE INDEX IF NOT EXISTS {name} ON image_meta({column})"
        print(statement)
        connection.execute(statement)
    connection.commit()
    connection.execute("pragma wal_checkpoint(TRUNCATE)")
    connection.execute("pragma optimize")


def image_id_for_path(relative_path: str) -> str:
    """画像IDは、画像ルートからの相対パス文字列のSHA-256。"""
    return hashlib.sha256(str(relative_path).encode()).hexdigest()


@contextmanager
def _temporary_id_table(connection: sqlite3.Connection, image_ids: list[str]):
    """指定した画像IDだけを持つ一時テーブルを作り、その名前を渡す。"""
    table = f"valid_id_{uuid.uuid4().hex}"
    cursor = connection.cursor()
    try:
        cursor.execute(f'CREATE TEMP TABLE "{table}" (image_id TEXT PRIMARY KEY)')
        try:
            for start in tqdm.tqdm(range(0, len(image_ids), _ID_CHUNK)):
                cursor.executemany(
                    f'INSERT INTO "{table}" (image_id) VALUES (?)',
                    ((image_id,) for image_id in image_ids[start:start + _ID_CHUNK]),
                )
        except Exception:
            connection.rollback()
            raise
        connection.commit()
        yield table
    finally:
        cursor.execute(f'DROP TABLE IF EXISTS "{table}"')


def find_images_without_embedding(
    connection: sqlite3.Connection, image_ids: list[str], dimension: int | None,
) -> list[str]:
    """埋め込みが未保存、またはdimension次元でない画像IDを返す。

    dimensionがNoneの場合は、次元を問わず保存済みの埋め込みを有効とみなす。
    埋め込み本体は読まず、その長さだけをSQLiteに問い合わせる。
    """
    if not image_ids:
        return []
    with _temporary_id_table(connection, image_ids) as table:
        rows = connection.execute(
            f"""
            SELECT valid_id.image_id
              FROM "{table}" AS valid_id
              LEFT JOIN image_meta ON valid_id.image_id = image_meta.image_id
             WHERE image_meta.meta IS NULL
                OR (? IS NOT NULL AND length(image_meta.meta) != ?)
            """,
            (dimension, None if dimension is None else dimension * FLOAT32_BYTES),
        )
        return [row[0] for row in rows]


class OrphanDbStats(NamedTuple):
    total_db_count: int
    # (image_id, image_path)
    orphan_records: list[tuple[str, str | None]]

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_records)


def collect_orphan_db_records(connection: sqlite3.Connection, image_ids: list[str]) -> OrphanDbStats:
    """データベースにだけ存在し、現在の画像ディレクトリにはない記録を集める。"""
    total_db_count = int(connection.execute("SELECT COUNT(*) FROM image_meta").fetchone()[0])
    if total_db_count == 0:
        return OrphanDbStats(total_db_count=0, orphan_records=[])
    with _temporary_id_table(connection, image_ids) as table:
        orphan_records = connection.execute(
            f"""
            SELECT image_meta.image_id, image_meta.image_path
              FROM image_meta
              LEFT JOIN "{table}" AS valid_id
                ON image_meta.image_id = valid_id.image_id
             WHERE valid_id.image_id IS NULL
            """
        ).fetchall()
    return OrphanDbStats(total_db_count=total_db_count, orphan_records=orphan_records)


def delete_orphan_db_records(connection: sqlite3.Connection, orphan_db_stats: OrphanDbStats) -> None:
    if orphan_db_stats.orphan_count == 0:
        print("No orphan DB records to delete.")
        return

    print("Deleting orphan DB records...")
    with connection:
        connection.executemany(
            "DELETE FROM image_meta WHERE image_id = ?",
            ((image_id,) for image_id, _ in orphan_db_stats.orphan_records),
        )

    print("Deleted records:")
    for image_id, image_path in orphan_db_stats.orphan_records:
        print(f"- {image_path if image_path else image_id}")


def infer_meta_dimension(connection: sqlite3.Connection) -> int | None:
    """保存済みの埋め込みの次元。埋め込みがなければNone。

    次元を変更した後に古い次元の記録が残っている場合など、複数の次元が混在していれば、
    どれに合わせるかを推測せずにエラーにする。
    """
    # 長さごとの件数は、長さだけを読んでここで数える。集計をSQLに任せると、SQLiteは埋め込みの本体まで
    # 並べ替え用の領域へ複製し、保存済みの埋め込み全体と同じ量のメモリを使う。
    sizes = sorted(Counter(
        size for (size,) in connection.execute("SELECT length(meta) FROM image_meta WHERE typeof(meta) = 'blob'")
    ).items())
    for size, _ in sizes:
        if size % FLOAT32_BYTES != 0:
            raise ValueError(f"Invalid meta byte size for float32 vector: {size}")
    if len(sizes) > 1:
        found = ", ".join(f"{size // FLOAT32_BYTES} ({count} rows)" for size, count in sizes)
        raise ValueError(
            f"Stored embeddings have different dimensions: {found}. "
            "Pass --search-model-out-dim to choose the dimension of this index. "
            "Records of images that no longer exist keep their old dimension; an update run with both "
            "--search-model-out-dim and --delete-orphan-db-records removes them."
        )
    return sizes[0][0] // FLOAT32_BYTES if sizes else None


def upsert_image_meta(connection: sqlite3.Connection, rows: Iterable[tuple]) -> None:
    connection.executemany(UPSERT_IMAGE_META, rows)
