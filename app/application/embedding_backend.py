from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image as PILImage


class SearchEmbeddingBackend(ABC):
    """検索クエリ生成用の埋め込みバックエンド共通インターフェイス。"""

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """検索テキストを ``np.float32`` の2次元特徴量配列に変換する。"""

        pass

    @abstractmethod
    def encode_image(self, image: PILImage.Image) -> np.ndarray:
        """検索画像を ``np.float32`` の2次元特徴量配列に変換する。"""

        pass


def to_float32_2d_array(features: Any) -> np.ndarray:
    """各バックエンドの出力を検索用の ``np.float32`` 2D配列に統一する。"""

    if hasattr(features, "to"):
        features = features.to("cpu")
    if hasattr(features, "detach"):
        features = features.detach()
    if hasattr(features, "numpy"):
        features = features.numpy()

    array = np.asarray(features, dtype=np.float32).copy()
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    return array.reshape(array.shape[0], -1)
