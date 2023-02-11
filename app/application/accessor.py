
from abc import abstractmethod
from typing import Any
import numpy as np

from app.domain.domain_object import ImageId, ImageItem, Model, ModelId, Tokenizer


class Accessor():
    """キャッシュなどドメイン(アプリケーションの役割)と直接関係ない外部接続を行うインターフェイス"""

    @abstractmethod
    def load_meta(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        """外部から対応したメタ情報を取得する"""

        pass

    @abstractmethod
    def load_index_file(self, id: ModelId) -> Any:
        """外部から対応したインデックスを読み込む"""

        pass

    @abstractmethod
    def load_model(self, id: ModelId) -> Model:

        pass

    @abstractmethod
    def load_tokenizer(self, id: ModelId) -> Tokenizer:

        pass

    @abstractmethod
    def load_index_item_list(self) -> list[ImageItem]:

        pass

    @abstractmethod
    def load_aesthetic_quality_list(self) -> dict[ImageId, float]:

        pass
