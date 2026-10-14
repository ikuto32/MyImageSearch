r"""Measure search over synthetic metadata: loading rows, text matching and ranking.

Run from the project root:
    uv run python scripts/benchmark_metadata.py

A synthetic database and a flat FAISS index are generated in a temporary
directory, then read through the same code as the application. No embedding
model, image file or network is used. Vector search uses an exact flat index,
so its time grows with the row count unlike the IVF index of a real dataset.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIMENSION = 16


def create_dataset(directory: Path, count: int):
    import faiss
    import numpy as np

    from app.infrastructure.index_files import IndexFiles

    files = IndexFiles(directory)
    rng = np.random.default_rng(20260906)
    vectors = rng.normal(size=(count, DIMENSION)).astype(np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(vectors)
    with files.index.open("wb") as stream:
        faiss.write_index(index, faiss.PyCallbackIOWriter(stream.write))
    width = len(str(count))
    ratings = ("general", "sensitive", "questionable", "explicit", None)
    with closing(sqlite3.connect(files.database)) as connection:
        connection.execute("""CREATE TABLE image_meta (
            image_id TEXT PRIMARY KEY, image_path TEXT, meta BLOB, image_tags TEXT,
            aesthetic_quality REAL, pony_aesthetic_quality REAL, style_cluster TEXT, rating TEXT)""")
        connection.executemany("INSERT INTO image_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            (
                f"{row:064x}", f"folder_{row // 1000:05d}/image_{row:0{width}d}.jpg", vectors[row].tobytes(),
                f"tag_{row % 97}, tag_{row % 13}, landscape", (row % 100) / 10, (row % 37) / 3.7,
                f"style_cluster_{row % 2048}", ratings[row % 5],
            )
            for row in range(count)
        ))
        connection.execute("CREATE INDEX idx_image_meta_path ON image_meta(image_path)")
        connection.commit()
    return files


def measure(operation, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - started) * 1000)
    return {"median_ms": round(statistics.median(samples), 2), "min_ms": round(min(samples), 2), "results": len(result.items)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=int, default=1_000_000)
    parser.add_argument("--results", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.items < 1 or args.results < 1 or args.repeats < 1:
        parser.error("items, results and repeats must be positive")

    import numpy as np

    from app.application.search import SearchOptions
    from app.application.usecase import Usecase
    from app.domain.domain_object import ModelId
    from app.domain.search_query import SearchQuery
    from app.infrastructure.search_index_loader import load_search_index

    model = ModelId("synthetic", "")
    with tempfile.TemporaryDirectory() as directory:
        files = create_dataset(Path(directory), args.items)
        tracemalloc.start()
        started = time.perf_counter()
        search_index = load_search_index(files)
        load_ms = (time.perf_counter() - started) * 1000
        retained, _peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    class Accessor:
        def load_search_index(self, _model_id):
            return search_index

    class Repository:
        def remember_image_items(self, _items):
            pass

    usecase = Usecase(Repository(), Accessor())
    options = SearchOptions(result_size=args.results)
    aesthetic = SearchOptions(0.3, (2.0, 8.0), "original", args.results)
    query = SearchQuery(tuple(np.random.default_rng(1).normal(size=DIMENSION).tolist()), False, model)
    report = {
        "scope": "Synthetic metadata through the application's loader and search; no embedding model",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "item_count": args.items,
        "result_limit": args.results,
        "load_index_and_metadata_ms": round(load_ms, 2),
        "metadata_python_retained_mb": round(retained / 1024 / 1024, 1),
        "name_substring_all_rows_match": measure(lambda: usecase.search_metadata(model, "name", "image", False, options), args.repeats),
        "name_substring_few_rows_match": measure(lambda: usecase.search_metadata(model, "name", "folder_00123/", False, options), args.repeats),
        "tags_regexp": measure(lambda: usecase.search_metadata(model, "tags", r"\btag_1\b", True, options), args.repeats),
        "tags_with_aesthetic_filter": measure(lambda: usecase.search_metadata(model, "tags", "tag_5", False, aesthetic), args.repeats),
        "vector_exact": measure(lambda: usecase.search_query(model, query, options), args.repeats),
        "vector_exact_with_aesthetic_filter": measure(lambda: usecase.search_query(model, query, aesthetic), args.repeats),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
