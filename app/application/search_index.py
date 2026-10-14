from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.domain.domain_object import ImageId, ImageItem

AESTHETIC_MODEL_NAMES = ("original", "pony")


@dataclass(frozen=True)
class SearchIndex:
    """FAISS indexと、その行番号順に並べたメタデータの列。

    画像1件ごとのオブジェクトを全件分保持せず列で持つことで、大規模データでも
    読み込み時間とメモリを抑える。画像項目は検索結果に含まれる行だけ生成する。
    """

    index: Any
    # 中心化していない旧形式のindexではNone。
    mean_vector: np.ndarray | None
    ids: Sequence[str]
    names: Sequence[str]
    tags: Sequence[str]
    ratings: Sequence[str]
    style_clusters: Sequence[str]
    # 審美性モデル名ごとの評価値。未評価の行はNaN。
    aesthetic_scores: Mapping[str, np.ndarray]
    # 行番号順が名前順でもある場合、同点の整列を行番号だけで行える。
    names_sorted: bool = False

    def __post_init__(self) -> None:
        columns = (self.ids, self.names, self.tags, self.ratings, self.style_clusters,
                   *self.aesthetic_scores.values())
        for column in columns:
            if len(column) != self.index.ntotal:
                raise ValueError(
                    f"Search index contains {self.index.ntotal} vectors, but metadata contains "
                    f"{len(column)} images. Rebuild the index and metadata together."
                )

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def items(self, positions: np.ndarray) -> list[ImageItem]:
        """行番号の配列から、その順序で画像項目を生成する。"""
        ids, names, tags = self.ids, self.names, self.tags
        ratings, style_clusters = self.ratings, self.style_clusters
        qualities = self.aesthetic_scores[AESTHETIC_MODEL_NAMES[0]][positions].tolist()
        return [
            ImageItem(
                ImageId(ids[position]), names[position], tags[position], ratings[position],
                # NaN(未評価)は自身と等しくない。
                style_clusters[position], quality if quality == quality else None,
            )
            for position, quality in zip(positions.tolist(), qualities)
        ]
