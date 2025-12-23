
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
    def load_index_file(self, model_id: ModelId) -> Any:
        """外部から対応したインデックスを読み込む"""

        pass

    @abstractmethod
    def load_model(self, model_id: ModelId) -> Model:

        pass

    @abstractmethod
    def load_tokenizer(self, model_id: ModelId) -> Tokenizer:

        pass

    @abstractmethod
    def load_index_item_list(self, model_id: ModelId) -> list[ImageItem]:

        pass

    @abstractmethod
    def load_aesthetic_quality_list(self, model_id: ModelId, aesthetic_model_name) -> dict[ImageId, float]:

        pass

    @abstractmethod
    def load_image_meta_info(self, model_id: ModelId, image_id: ImageId) -> dict[str, str]:
        """画像メタ情報を取得する"""

        pass

    @abstractmethod
    def load_style_cluster_list(self, model_id: ModelId) -> dict[ImageId, str]:
        """style_cluster 情報を取得する"""

        pass

    @abstractmethod
    def load_rating_list(self, model_id: ModelId) -> dict[ImageId, str]:
        """rating 情報を取得する"""

        pass
