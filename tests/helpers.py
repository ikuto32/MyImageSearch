"""Fixtures shared by the test modules: small catalogs, images and search indexes."""
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image as PILImage

from app.application.search_index import SearchIndex
from app.infrastructure.index_files import IndexFiles, open_readonly_database

CATALOG_COLUMNS = ("image_id", "image_path", "image_tags", "rating")
COLUMN_TYPES = {
    "image_id": "TEXT PRIMARY KEY", "meta": "BLOB",
    "aesthetic_quality": "REAL", "pony_aesthetic_quality": "REAL",
}


def create_catalog(meta_dir: Path, name: str, rows, columns=CATALOG_COLUMNS, *, path_index=True) -> IndexFiles:
    """Create a registered index directory whose FAISS file must never be read."""
    files = IndexFiles(meta_dir / name)
    files.directory.mkdir(parents=True)
    files.index.write_bytes(b"not a FAISS index; catalog queries must not read it")
    with closing(sqlite3.connect(files.database)) as connection:
        declared = ", ".join(f"{column} {COLUMN_TYPES.get(column, 'TEXT')}" for column in columns)
        connection.execute(f"CREATE TABLE image_meta ({declared})")
        if path_index:
            connection.execute("CREATE INDEX idx_image_meta_path ON image_meta(image_path)")
        marks = ",".join("?" for _ in columns)
        connection.executemany(f"INSERT INTO image_meta VALUES ({marks})", rows)
        connection.commit()
    return files


def traced_connections(queries: list):
    """A replacement for open_readonly_database that records every executed statement."""
    def connect(path):
        connection = open_readonly_database(path)
        connection.set_trace_callback(queries.append)
        return connection
    return connect


def save_image(path: Path, size=(20, 20), color="red", **options) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", size, color).save(path, **options)
    return path


def make_search_index(index, names, *, ids=None, mean_vector=None, tags=None, ratings=None,
                      style_clusters=None, original=None, pony=None, names_sorted=None) -> SearchIndex:
    """A SearchIndex whose image IDs equal their names unless given; unset scores are unrated."""
    count = len(names)

    def scores(values):
        return np.array([np.nan] * count if values is None else values, dtype=np.float64)

    return SearchIndex(
        index=index,
        mean_vector=mean_vector,
        ids=list(names if ids is None else ids),
        names=list(names),
        tags=list(tags or [""] * count),
        ratings=list(ratings or [""] * count),
        style_clusters=list(style_clusters or [""] * count),
        aesthetic_scores={"original": scores(original), "pony": scores(pony)},
        names_sorted=list(names) == sorted(names) if names_sorted is None else names_sorted,
    )
