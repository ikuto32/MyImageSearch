"""画像から埋め込み・審美性評価・スタイルクラスタ・タグを求め、SQLiteへ保存する。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import datetime
import math
import sqlite3
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import tqdm

from indexer.database import FLOAT32_BYTES, upsert_image_meta
from indexer.embedding import OpenClipEmbeddingBackend, SearchEmbeddingBackend
from indexer.scoring import AestheticModel, PonyAestheticScorer, StyleCluster
from indexer.tagging import ImageTaggingService, load_sidecar_tags

# コミットの間隔(画像枚数の目安)。
_COMMIT_INTERVAL_IMAGES = 10_000
_OPTIMIZE_INTERVAL_ROWS = 1_000_000


@dataclass(frozen=True)
class MetadataModels:
    """OpenCLIP ViT-L/14の特徴量から、審美性評価とスタイルクラスタを求める一式。"""

    aesthetic: AestheticModel
    pony: PonyAestheticScorer
    style: StyleCluster

    def score(self, internal: torch.Tensor, projected: torch.Tensor) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """(審美性評価, Pony審美性評価, スタイルクラスタID)をバッチの順で返す。

        射影後の特徴量を審美性評価に、射影前の内部特徴量をスタイルクラスタに使う。
        """
        features = _l2_normalize(projected)
        aesthetic_dtype = next(self.aesthetic.parameters()).dtype
        aesthetic = self.aesthetic.score_batch(features.to(dtype=aesthetic_dtype))
        cluster_ids, _distances = self.style.get_cluster_batch(internal)
        return aesthetic.detach().cpu().numpy().reshape(-1), self.pony.score_batch(features), cluster_ids


def _l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return (features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)).contiguous()


def _to_device(batch, device):
    """Recursively move tensors and mapping-like processor outputs to a device."""

    if batch is None:
        return None
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: _to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_to_device(value, device) for value in batch]
    return batch


def _to_numpy(batch) -> np.ndarray:
    return batch.cpu().numpy() if isinstance(batch, torch.Tensor) else np.asarray(batch)


class FeatureExtractor:
    """DataLoaderのバッチを推論し、画像ごとの行としてデータベースへ保存する。"""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        device: str,
        search_model: SearchEmbeddingBackend,
        search_model_label: str,
        metadata_models: MetadataModels | None,
        metadata_embedding: OpenClipEmbeddingBackend | None,
        metadata_model_label: str | None,
        tagging_service: ImageTaggingService | None,
        image_dir: str,
        dimension: int | None,
    ):
        self.connection = connection
        self.device = device
        self.search_model = search_model
        self.search_model_label = search_model_label
        # Noneの場合は、審美性評価とスタイルクラスタを保存しない。
        self.metadata_models = metadata_models
        # メタデータ用の特徴量を求めるモデル。Noneの場合は、検索モデルの特徴量をそのまま使う。
        self.metadata_embedding = metadata_embedding
        self.metadata_model_label = metadata_model_label
        # Noneの場合は、画像の隣のタグファイルを読む。
        self.tagging_service = tagging_service
        self.image_dir = image_dir
        # 保存する埋め込みの次元。Noneなら最初の埋め込みから決める。
        self.dimension = dimension
        self.stored_count = 0

    def run(
        self,
        loader,
        image_paths: list[str],
        image_ids: list[str],
        seen: np.ndarray,
        *,
        batch_size: int,
    ) -> None:
        """loaderが返す画像を処理する。処理に進んだ画像は、成否によらずseenに記録する。

        バッチ単位の失敗は記録して次へ進む。loader自体の失敗は呼び出し側へ伝わる。
        """
        commit_interval = max(1, _COMMIT_INTERVAL_IMAGES // batch_size)
        stored_batches = 0
        try:
            with torch.inference_mode():
                for batch in tqdm.tqdm(loader, total=math.ceil(len(image_paths) / batch_size)):
                    if batch is None:
                        continue
                    indexes = [int(index) for index in batch[2]]
                    # 転送の失敗(メモリ不足など)はバッチの失敗として扱わず、呼び出し側の再試行に任せる。
                    search_input = _to_device(batch[0], self.device)
                    metadata_input = _to_device(batch[1], self.device)
                    seen[indexes] = True
                    try:
                        rows = self._rows(
                            search_input, metadata_input, batch[3], batch[4], batch[5], indexes, image_paths, image_ids,
                        )
                        if not rows:
                            continue
                        upsert_image_meta(self.connection, rows)
                    except Exception:
                        traceback.print_exc()
                        continue
                    self.stored_count += len(rows)
                    stored_batches += 1
                    if stored_batches % commit_interval == 0:
                        print(f"commit after batch {stored_batches} (total rows: {self.stored_count})")
                        self.connection.commit()
        finally:
            # loaderが途中で失敗しても、そこまでの結果は残す。
            self.connection.commit()

    def _rows(
        self, search_input, metadata_input, wd_input, z3d_input, file_hashes: list[str],
        indexes: list[int], image_paths: list[str], image_ids: list[str],
    ) -> list[tuple]:
        internal, search_features = self.search_model.encode_image_with_internal(search_input)
        embeddings = _l2_normalize(search_features).detach().cpu().float().numpy()

        aesthetic_scores = pony_scores = cluster_ids = [None] * len(embeddings)
        if self.metadata_models is not None:
            projected = search_features
            if self.metadata_embedding is not None:
                internal, projected = self.metadata_embedding.encode_image_with_internal(metadata_input)
            aesthetic_scores, pony_scores, cluster_ids = self.metadata_models.score(internal, projected)

        if self.tagging_service is None:
            tags = [load_sidecar_tags(self.image_dir, image_paths[index]) for index in indexes]
        else:
            tags = self.tagging_service.tag_batch(_to_numpy(wd_input), _to_numpy(z3d_input))

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rows = []
        for embedding, index, aesthetic, pony, cluster_id, tag, file_hash in zip(
            embeddings, indexes, aesthetic_scores, pony_scores, cluster_ids, tags, file_hashes
        ):
            if self.dimension is None:
                self.dimension = int(embedding.shape[0])
                print(f"Detected embedding dimension: {self.dimension}")
            if embedding.shape[0] != self.dimension:
                print(f"{image_paths[index]}: embedding has {embedding.shape[0]} dimensions, expected {self.dimension}")
                continue
            rows.append((
                image_ids[index],
                image_paths[index],
                embedding.tobytes(),
                None if aesthetic is None else float(aesthetic),
                None if pony is None else float(pony),
                "" if cluster_id is None else f"style_cluster_{cluster_id}",
                tag.get("rating", ""),
                ", ".join(tag.get("tags", [])),
                timestamp,
                self.search_model_label,
                self.metadata_model_label,
                file_hash,
            ))
        return rows


# =====================================================
# 保存済みの埋め込みからの審美性評価の再計算

class SQLiteMetaDataset(IterableDataset):
    """rowidの範囲をワーカーごとに分け、(image_id, 埋め込み)を読む。"""

    def __init__(self, db_path: str, expected_bytes: int, min_rowid: int, max_rowid: int, fetch_size: int = 10000):
        self.db_path = db_path
        self.expected_bytes = expected_bytes
        self.min_rowid = min_rowid
        self.max_rowid = max_rowid
        self.fetch_size = fetch_size

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            start_row, end_row = self.min_rowid, self.max_rowid
        else:
            total_range = self.max_rowid - self.min_rowid + 1
            chunk_size = math.ceil(total_range / worker_info.num_workers)
            start_row = self.min_rowid + worker_info.id * chunk_size
            end_row = min(start_row + chunk_size - 1, self.max_rowid)

        connection = sqlite3.connect(self.db_path, timeout=60.0)
        try:
            # 読み取りロックを長く保持しないよう、fetch_size行ずつ読み切る。
            current_min = start_row
            while current_min <= end_row:
                rows = connection.execute(
                    """
                    SELECT rowid, image_id, meta
                    FROM image_meta
                    WHERE rowid BETWEEN ? AND ?
                      AND meta IS NOT NULL
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (current_min, end_row, self.fetch_size),
                ).fetchall()
                if not rows:
                    break
                for rowid, image_id, meta_blob in rows:
                    current_min = rowid + 1
                    if len(meta_blob) != self.expected_bytes:
                        yield None, None
                        continue
                    yield image_id, torch.from_numpy(np.frombuffer(meta_blob, dtype=np.float32).copy())
        finally:
            connection.close()


def _meta_collate_fn(batch):
    """次元の合わない行を除いてテンソルにまとめ、読んだ行数とともに返す"""
    valid = [(image_id, vector) for image_id, vector in batch if image_id is not None]
    if not valid:
        return [], torch.empty(0), len(batch)
    image_ids, vectors = zip(*valid)
    return list(image_ids), torch.stack(vectors), len(batch)


def recalc_aesthetic_scores(
    connection: sqlite3.Connection,
    aesthetic_model: AestheticModel,
    pony_scorer: PonyAestheticScorer,
    device: str,
    dimension: int,
    batch_size: int = 16384,
    num_workers: int = 8,
) -> int:
    """保存済みの埋め込みを並列に読み、審美性評価だけを計算し直して更新する。"""
    db_path = connection.execute("PRAGMA database_list").fetchall()[0][2]
    if not db_path:
        raise ValueError("[recalc_aesthetic] DataLoader parallelization requires a physical database file.")
    total, min_rowid, max_rowid = connection.execute(
        "SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM image_meta"
    ).fetchone()
    print(f"[recalc_aesthetic] rows: {total} (workers: {num_workers})")
    if total == 0:
        return 0

    dataloader = DataLoader(
        SQLiteMetaDataset(db_path, dimension * FLOAT32_BYTES, min_rowid, max_rowid, fetch_size=batch_size * 4),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_meta_collate_fn,
        prefetch_factor=16 if num_workers > 0 else None,
    )

    updated = 0
    aesthetic_dtype = next(aesthetic_model.parameters()).dtype
    with tqdm.tqdm(total=total, unit="row", desc="recalc_aesthetic") as progress:
        for image_ids, features, read_count in dataloader:
            progress.update(read_count)
            if not image_ids:
                continue

            features = _l2_normalize(features.to(device=device, dtype=aesthetic_dtype))
            aesthetic_scores = aesthetic_model.score_batch(features).cpu().numpy().reshape(-1)
            pony_scores = pony_scorer.score_batch(features).reshape(-1)
            connection.executemany(
                """
                UPDATE image_meta
                   SET aesthetic_quality      = ?,
                       pony_aesthetic_quality = ?
                 WHERE image_id = ?
                """,
                [
                    (float(aesthetic), float(pony), image_id)
                    for image_id, aesthetic, pony in zip(image_ids, aesthetic_scores, pony_scores)
                ],
            )
            connection.commit()

            previous = updated
            updated += len(image_ids)
            if updated // _OPTIMIZE_INTERVAL_ROWS > previous // _OPTIMIZE_INTERVAL_ROWS:
                # busy=1 なら競合してチェックポイントに失敗している
                result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                print(f"\n[recalc_aesthetic] updated {updated} rows; checkpoint (busy, log, checkpointed): {result}")
                connection.execute("PRAGMA optimize")

    print(f"[recalc_aesthetic] updated {updated} rows")
    return updated
