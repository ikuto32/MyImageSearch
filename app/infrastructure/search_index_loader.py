"""FAISS index・平均ベクトル・行メタデータを、同じ作成世代のものとして読み込む。"""
from contextlib import closing
import hashlib
from itertools import islice
import logging
import operator
import sqlite3
from typing import Any

import faiss
import numpy as np

from app.application.search_index import SearchIndex
from app.infrastructure.index_files import (
    IndexFiles,
    IndexManifest,
    image_meta_columns,
    open_readonly_database,
    read_index_manifest,
    readonly_database_uri,
    sidecar_requires_manifest,
)

logger = logging.getLogger(__name__)

_FETCH_ROWS = 65536
_READ_BYTES = 1024 * 1024


def load_search_index(files: IndexFiles) -> SearchIndex:
    with closing(open_readonly_database(files.database)) as connection:
        has_manifest = files.manifest.is_file()
        if has_manifest:
            connection.execute("ATTACH DATABASE ? AS manifest", (readonly_database_uri(files.manifest),))
        # The header and row mapping come from one attached SQLite snapshot,
        # even if a new manifest is published concurrently.
        connection.execute("BEGIN")
        manifest = (
            read_index_manifest(files.manifest, connection=connection, schema="manifest")
            if has_manifest else None
        )
        if manifest is None and sidecar_requires_manifest(files.sidecar):
            raise ValueError("Index row manifest is missing. Rebuild the index.")

        index, index_sha256 = _read_faiss_index(files)
        if manifest is not None:
            if index_sha256 != manifest.index_sha256:
                raise ValueError("Index file and row manifest belong to different builds. Rebuild the index.")
            if manifest.dimension != index.d or manifest.count != index.ntotal:
                raise ValueError("Index manifest dimension/count does not match FAISS. Rebuild the index.")
        mean_vector = _read_mean_vector(files, manifest, int(index.d))
        if manifest is None and (files.manifest.is_file() or sidecar_requires_manifest(files.sidecar)):
            # チェックサムのない旧形式を読んでいる間に、新しい形式のindexが公開された。
            raise ValueError("Index was rebuilt while it was being loaded. Retry the search.")
        columns = _read_rows(connection, manifest, int(index.d))

    ids, names, tags, ratings, style_clusters, original, pony = columns
    return SearchIndex(
        index=index,
        mean_vector=mean_vector,
        ids=ids,
        names=names,
        tags=tags,
        ratings=ratings,
        style_clusters=style_clusters,
        aesthetic_scores={
            # NULL(未評価)はNaNになる。
            "original": np.array(original, dtype=np.float64),
            "pony": np.array(pony, dtype=np.float64),
        },
        names_sorted=all(map(operator.le, names, islice(names, 1, None))),
    )


def _read_faiss_index(files: IndexFiles) -> tuple[Any, str]:
    """Hash the same file handle FAISS reads, avoiding a second pass over a potentially huge file."""
    digest = hashlib.sha256()
    with files.index.open("rb") as stream:
        def read_bytes(size):
            data = stream.read(size)
            digest.update(data)
            return data
        index = faiss.read_index(faiss.PyCallbackIOReader(read_bytes))
        while read_bytes(_READ_BYTES):
            pass
    return index, digest.hexdigest()


def _read_mean_vector(files: IndexFiles, manifest: IndexManifest | None, dimension: int) -> np.ndarray | None:
    if not files.mean.is_file():
        if manifest is not None or files.sidecar.is_file():
            raise ValueError("Mean vector is missing from a centered index. Rebuild the index.")
        logger.warning(
            "旧形式の非中心化FAISS indexとして検索します（mean/metadata sidecarなし）: %s", files.index
        )
        return None
    with files.mean.open("rb") as stream:
        if manifest is not None:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest.mean_sha256:
                raise ValueError("Mean vector and index manifest belong to different builds. Rebuild the index.")
            stream.seek(0)
        mean_vector = np.load(stream, allow_pickle=False)
    if mean_vector.ndim != 1 or not np.isfinite(mean_vector).all():
        raise ValueError("Mean vector must be a finite one-dimensional array. Rebuild the index.")
    if mean_vector.size != dimension:
        raise ValueError("Mean vector dimension does not match the index. Rebuild the index.")
    return mean_vector


def _read_rows(
    connection: sqlite3.Connection, manifest: IndexManifest | None, dimension: int,
) -> tuple[list, ...]:
    """FAISSの行番号と同じ順序で、メタデータを列ごとのリストとして読む。"""
    available = image_meta_columns(connection)
    prefix = "m." if manifest is not None else ""

    def text(name: str) -> str:
        return f"COALESCE({prefix}{name}, '')" if name in available else "''"

    def real(name: str) -> str:
        return f"{prefix}{name}" if name in available else "NULL"

    fields = ", ".join((
        f"{prefix}image_id", text("image_path"), text("image_tags"), text("rating"),
        text("style_cluster"), real("aesthetic_quality"), real("pony_aesthetic_quality"),
    ))
    if manifest is not None:
        invalid_rows = connection.execute("""
            SELECT COUNT(*) FROM manifest.index_rows AS r
            LEFT JOIN image_meta AS m ON m.image_id = r.image_id
            WHERE m.image_id IS NULL OR m.image_path IS NOT r.image_path
        """).fetchone()[0]
        if invalid_rows:
            raise ValueError("Index manifest references missing or changed image paths. Rebuild the index.")
        cursor = connection.execute(f"""
            SELECT {fields}
            FROM manifest.index_rows AS r JOIN image_meta AS m ON m.image_id = r.image_id
            ORDER BY r.position
        """)
    else:
        # Legacy builders added only float32 vectors of the index dimension,
        # in SQLite image_path order. NULL/other-dimension rows were skipped.
        cursor = connection.execute(f"""
            SELECT {fields}
            FROM image_meta
            WHERE typeof(meta) = 'blob' AND length(meta) = ?
            ORDER BY image_path
        """, (dimension * 4,))

    columns: tuple[list, ...] = tuple([] for _ in range(7))
    while rows := cursor.fetchmany(_FETCH_ROWS):
        for column, values in zip(columns, zip(*rows)):
            column.extend(values)
    return columns
