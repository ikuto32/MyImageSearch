"""SQLiteに保存した埋め込みから、中心化したFAISS indexとその行マニフェストを作る。"""
from __future__ import annotations

from itertools import chain, islice
import pathlib
import sqlite3
import time
import uuid

import faiss
import numpy as np
import tqdm

from app.infrastructure.index_files import (
    IndexFiles,
    IndexManifest,
    create_manifest_tables,
    file_sha256,
    write_index_sidecar,
    write_manifest_header,
)
from indexer.database import FLOAT32_BYTES, connect_db

# indexに追加する行の条件。行マニフェストのない旧形式を検索側が読むときも同じ条件を使う。
_INDEXED_ROWS = "FROM image_meta WHERE typeof(meta) = 'blob' AND length(meta) = ?"
_SAMPLING_SEED = 42
# 公開するファイルを検索アプリが読み込んでいる間、置き換えを待つ秒数と、その確認の間隔。
_PUBLISH_TIMEOUT_SEC = 120
_PUBLISH_RETRY_SEC = 0.2


def create_ivfpq_index(n_centroids: int, pq_m: int, bits_per_code: int, dim: int) -> faiss.Index:
    quantizer = faiss.IndexFlatIP(dim)
    return faiss.IndexIVFPQ(quantizer, dim, n_centroids, pq_m, bits_per_code, faiss.METRIC_INNER_PRODUCT)


def _vectors(blobs, dimension: int) -> np.ndarray:
    """同じ次元のfloat32 BLOBを、書き込み可能な(N, dimension)配列にまとめる。"""
    return np.frombuffer(bytearray().join(blobs), dtype=np.float32).reshape(-1, dimension)


def _fetch_batches(cursor: sqlite3.Cursor, batch_size: int):
    while rows := cursor.fetchmany(batch_size):
        yield rows


def compute_mean_vector(connection: sqlite3.Connection, dimension: int, batch_size: int = 50_000) -> np.ndarray:
    """全ベクトルをL2正規化してから平均ベクトルを計算する。

    indexと検索クエリは正規化の後に中心化するため、平均も正規化後のベクトルから求める。
    """
    print("Computing mean vector...")
    cursor = connection.execute(f"SELECT meta {_INDEXED_ROWS}", (dimension * FLOAT32_BYTES,))
    # float64 で累積して精度を保つ
    total = np.zeros(dimension, dtype=np.float64)
    count = 0
    with tqdm.tqdm(desc="mean computation") as progress:
        for rows in _fetch_batches(cursor, batch_size):
            vectors = _vectors((row[0] for row in rows), dimension)
            faiss.normalize_L2(vectors)
            total += vectors.sum(axis=0, dtype=np.float64)
            count += len(vectors)
            progress.update(len(vectors))

    if count == 0:
        raise ValueError("No valid meta vectors found.")
    return (total / count).astype(np.float32)


def collect_train_samples(
    connection: sqlite3.Connection,
    dimension: int,
    train_samples: int,
    batch_size: int,
    total: int,
) -> np.ndarray:
    """image_path順の走査から、最大train_samples件のベクトルを一様に抽出する。

    Algorithm L (Vitter 1985) による読み飛ばし式のreservoir sampling。
    乱数の種を固定しているため、同じデータベースからは同じ標本が得られる。
    """
    k = min(train_samples, total)
    reservoir = np.empty((max(k, 0), dimension), dtype=np.float32)
    if k <= 0:
        return reservoir
    cursor = connection.execute(
        f"SELECT meta {_INDEXED_ROWS} ORDER BY image_path", (dimension * FLOAT32_BYTES,)
    )

    # 先頭 k 件でリザーバを満たす
    filled = 0
    unread: list = []
    while filled < k:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return reservoir[:filled]
        taken = min(len(rows), k - filled)
        reservoir[filled:filled + taken] = _vectors((row[0] for row in rows[:taken]), dimension)
        filled += taken
        unread = rows[taken:]

    # 以降は、置き換える行までの間隔を乱数で決めて読み飛ばす
    rng = np.random.default_rng(_SAMPLING_SEED)
    remaining = chain(unread, chain.from_iterable(_fetch_batches(cursor, batch_size)))
    w = np.exp(np.log(rng.random()) / k)
    while True:
        skip = int(np.floor(np.log(rng.random()) / np.log(1.0 - w)))
        row = next(islice(remaining, skip, None), None)
        if row is None:
            return reservoir
        reservoir[rng.integers(0, k)] = np.frombuffer(row[0], dtype=np.float32)
        w *= np.exp(np.log(rng.random()) / k)


def _publish(temporary: pathlib.Path, target: pathlib.Path, deadline: float) -> None:
    """用意したファイルで公開先を置き換える。

    Windowsでは、ほかのプロセスが開いているファイルを置き換えられない。検索アプリは
    indexの読み込み中だけファイルを開いているため、読み終わるのを期限まで待つ。
    """
    waiting = False
    while True:
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                print(f"{target} is in use or not writable. Stop the search app and run the indexer again.")
                raise
            if not waiting:
                waiting = True
                print(f"waiting until {target.name} is no longer in use...")
            time.sleep(_PUBLISH_RETRY_SEC)


def build_index(
    files: IndexFiles,
    nlist: int,
    pq_m: int,
    bits_per_code: int,
    dimension: int,
    batch_size: int = 100_000,
    train_samples: int = 2_000_000,
) -> None:
    """Build a centered index plus a manifest of the exact image IDs added to it.

    Files are prepared beside the output and published only after the build
    succeeds. The manifest binds index/mean checksums and is published last;
    readers reject mixed generations if a build is interrupted during publication.
    """
    token = uuid.uuid4().hex
    published = (files.sidecar, files.mean, files.index, files.manifest)
    temporary = {
        target: files.index.with_name(f".{files.index.name}.{token}{suffix}")
        for target, suffix in zip(published, (".meta.json", ".mean.npy", ".index.tmp", ".rows.sqlite3"))
    }
    connection = connect_db(files.directory)
    manifest_connection = None
    try:
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("BEGIN")  # Train, mean, vectors and row IDs use one SQLite snapshot.

        total = connection.execute(
            f"SELECT COUNT(*) {_INDEXED_ROWS}", (dimension * FLOAT32_BYTES,)
        ).fetchone()[0]
        print(f"total rows with {dimension}-dimensional float32 meta: {total}")
        if total == 0:
            print("No valid meta found, skipping index creation.")
            return

        mean_vector = compute_mean_vector(connection, dimension, batch_size)
        if not np.isfinite(mean_vector).all():
            raise ValueError("Image embeddings contain non-finite values; index was not replaced.")
        with temporary[files.mean].open("wb") as stream:
            np.save(stream, mean_vector)
        write_index_sidecar(temporary[files.sidecar], embedding_dim=dimension, pq_m=pq_m)

        manifest_connection = sqlite3.connect(temporary[files.manifest])
        create_manifest_tables(manifest_connection)

        # -------- パス1: 訓練 --------
        train_vectors = collect_train_samples(connection, dimension, train_samples, batch_size, total)
        train_vectors = np.ascontiguousarray(train_vectors, dtype=np.float32)
        # ① 正規化 → ② 中心化(再正規化しない)
        faiss.normalize_L2(train_vectors)
        train_vectors -= mean_vector

        index = create_ivfpq_index(nlist, pq_m, bits_per_code, dimension)
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(f"Index must be IP metric, got {index.metric_type}")
        print(f"train {train_vectors.shape}")
        index.train(train_vectors)
        del train_vectors

        # -------- パス2: 逐次add --------
        print("add")
        cursor = connection.execute(
            f"SELECT image_id, image_path, meta {_INDEXED_ROWS} ORDER BY image_path, image_id",
            (dimension * FLOAT32_BYTES,),
        )
        with tqdm.tqdm(total=total, unit="row") as progress:
            for rows in _fetch_batches(cursor, batch_size):
                vectors = _vectors((row[2] for row in rows), dimension)
                faiss.normalize_L2(vectors)
                vectors -= mean_vector
                first_position = int(index.ntotal)
                index.add(vectors)
                manifest_connection.executemany(
                    "INSERT INTO index_rows VALUES (?, ?, ?)",
                    ((first_position + offset, row[0], row[1]) for offset, row in enumerate(rows)),
                )
                progress.update(len(rows))
        if index.ntotal != total:
            raise ValueError(f"Expected {total} vectors, added {index.ntotal}; index was not replaced.")

        print("write")
        # Japanese and other non-ASCII directory names must work on Windows.
        with temporary[files.index].open("wb") as stream:
            faiss.write_index(index, faiss.PyCallbackIOWriter(stream.write))
        write_manifest_header(manifest_connection, IndexManifest(
            count=int(index.ntotal),
            dimension=int(index.d),
            index_sha256=file_sha256(temporary[files.index]),
            mean_sha256=file_sha256(temporary[files.mean]),
        ))
        manifest_connection.commit()
        manifest_connection.close()
        manifest_connection = None
        # Publish the format marker first so a partial new build cannot be
        # mistaken for an old index without a manifest.
        deadline = time.monotonic() + _PUBLISH_TIMEOUT_SEC
        # 先に、公開先のどれも使用中でなくなるのを待つ。自分自身への置き換えは、ほかのプロセスが削除共有なしで
        # 開いている間だけ失敗する(検索アプリのopen()やSQLiteはこれに当たる)。待ち切れずに中断しても、
        # 公開済みのindexは元のまま使える。削除共有付きで開く他のソフトウェアは検出できず、その場合は
        # 一部だけ置き換わることがあるが、行マニフェストの照合で読み込みが拒否されるため、再実行すればよい。
        for target in published:
            if target.exists():
                _publish(target, target, deadline)
        for target in published:
            _publish(temporary[target], target, deadline)
    finally:
        if manifest_connection is not None:
            manifest_connection.close()
        print("close db connection")
        connection.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
