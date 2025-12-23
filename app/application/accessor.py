
from abc import abstractmethod
from typing import Any
import numpy as np

from app.domain.domain_object import ImageId, ImageItem, Model, ModelId, Tokenizer


class Accessor():
    """キャッシュなどドメイン(アプリケーションの役割)と直接関係ない外部接続を行うインターフェイス"""

    @abstractmethod
    def load_image_feature(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        """外部から対応したメタ情報を取得する"""

        pass

    @abstractmethod
    def load_model(self, model_id: ModelId) -> Model:

        pass

    @abstractmethod
    def load_tokenizer(self, model_id: ModelId) -> Tokenizer:

        pass

    @abstractmethod
    def load_index_with_metadata(
        self, model_id: ModelId, aesthetic_model_name: str
    ) -> tuple[Any, list[ImageItem]]:
        """外部から対応したインデックスとメタ情報付きの画像一覧を読み込む"""

        pass
