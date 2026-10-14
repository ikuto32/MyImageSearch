"""SearchIndexに対する近傍検索・審美性評価の適用・順位付け。"""
from dataclasses import dataclass

import faiss
import numpy as np

from app.application.search_index import AESTHETIC_MODEL_NAMES, SearchIndex
from app.domain import limits
from app.domain.domain_object import SearchResult

# IVF indexで走査するクラスタ数。
NPROBE = 64

AESTHETIC_QUALITY_MIN = 0.0
AESTHETIC_QUALITY_MAX = 10.0


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """すべての検索方法に共通する設定。"""

    # 審美性評価を類似度に加重する係数。0で無効。
    aesthetic_quality_beta: float = 0.0
    aesthetic_quality_range: tuple[float, float] = (AESTHETIC_QUALITY_MIN, AESTHETIC_QUALITY_MAX)
    aesthetic_model_name: str = AESTHETIC_MODEL_NAMES[0]
    result_size: int = limits.DEFAULT_RESULT_SIZE

    @property
    def uses_aesthetics(self) -> bool:
        """有効な場合、未評価の画像と範囲外の画像は結果から除かれる。"""
        low, high = self.aesthetic_quality_range
        return (
            self.aesthetic_quality_beta != 0
            or low > AESTHETIC_QUALITY_MIN
            or high < AESTHETIC_QUALITY_MAX
        )


def check_dimension(search_index: SearchIndex, features: np.ndarray, *, error_type=ValueError) -> None:
    """クエリ次元とインデックス次元が一致することを検証する。"""
    actual = int(features.shape[-1])
    if actual != search_index.dimension:
        raise error_type(
            f"Query embedding dimension mismatch: query has {actual} dimensions, "
            f"but the FAISS index expects {search_index.dimension}."
        )


def _eligible_rows(search_index: SearchIndex, options: SearchOptions) -> np.ndarray:
    # NaN(未評価)との比較は偽になり、範囲外と同じく除外される。
    quality = search_index.aesthetic_scores[options.aesthetic_model_name]
    low, high = options.aesthetic_quality_range
    return (quality >= low) & (quality <= high)


def nearest_neighbors(
    search_index: SearchIndex,
    features: np.ndarray,
    *,
    mean_centering: bool,
    options: SearchOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """クエリに近い行番号と内積スコアを返す。

    審美性評価の範囲はFAISSの検索時に適用するため、件数上限で対象が欠けることはない。
    """
    check_dimension(search_index, features)
    query = np.array(features, dtype=np.float32).reshape(1, -1)
    # indexの作成時と同じく、L2正規化の後に中心化し、再正規化はしない。
    faiss.normalize_L2(query)
    if mean_centering and search_index.mean_vector is not None:
        query -= search_index.mean_vector.astype(np.float32, copy=False)

    limit = min(options.result_size, len(search_index))
    selector = None
    if options.uses_aesthetics:
        eligible = _eligible_rows(search_index, options)
        limit = min(limit, int(np.count_nonzero(eligible)))
        selector = faiss.IDSelectorBitmap(np.packbits(eligible, bitorder="little"))
    if limit == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    index = search_index.index
    if hasattr(index, "nprobe"):
        parameters = (
            faiss.SearchParametersIVF(nprobe=NPROBE) if selector is None
            else faiss.SearchParametersIVF(nprobe=NPROBE, sel=selector)
        )
    else:
        parameters = None if selector is None else faiss.SearchParameters(sel=selector)
    distances, positions = index.search(query, limit, params=parameters)
    # FAISS は埋まらない枠を -1 で返す。
    found = positions[0] != -1
    return positions[0][found], distances[0][found].astype(np.float64)


def apply_aesthetics(
    search_index: SearchIndex,
    positions: np.ndarray,
    scores: np.ndarray,
    options: SearchOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """審美性評価が範囲外・未評価の行を除き、スコアに評価値を加重する。"""
    if not options.uses_aesthetics:
        return positions, scores
    quality = search_index.aesthetic_scores[options.aesthetic_model_name][positions]
    low, high = options.aesthetic_quality_range
    keep = (quality >= low) & (quality <= high)
    beta = options.aesthetic_quality_beta
    return positions[keep], scores[keep] * (1 - beta**2) + quality[keep] * beta


def rank_results(
    search_index: SearchIndex,
    positions: np.ndarray,
    scores: np.ndarray,
    limit: int,
) -> list[SearchResult]:
    """スコア降順、同点は名前昇順に並べ、先頭のlimit件だけ画像項目を生成する。"""
    if search_index.names_sorted:
        order = np.lexsort((positions, -scores))[:limit]
    else:
        names = search_index.names
        keys = zip((-scores).tolist(), (names[position] for position in positions.tolist()))
        order = np.array(
            sorted(range(len(positions)), key=list(keys).__getitem__)[:limit], dtype=np.int64,
        )
    return [
        SearchResult(item, score)
        for item, score in zip(search_index.items(positions[order]), scores[order].tolist())
    ]
