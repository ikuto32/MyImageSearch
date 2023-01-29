
from abc import abstractmethod
from typing import Any
import numpy as np

from app.domain.domain_object import ImageId, ModelId

class Accessor():
    """キャッシュなどドメイン(アプリケーションの役割)と直接関係ない外部接続を行うインターフェイス"""

    @abstractmethod
    def load_meta(self, id : ImageId) -> np.ndarray:
        """外部から対応したメタ情報を取得する"""

        pass


    @abstractmethod
    def load_Index_file(self, id : ModelId) -> Any :
        """外部から対応したインデックスを読み込む"""

        pass