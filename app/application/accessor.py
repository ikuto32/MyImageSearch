
from abc import abstractmethod
from typing import Any
import numpy as np

from app.domain.domain_object import ItemId

class Accessor():
    """キャッシュなどドメイン(アプリケーションの役割)と直接関係ない外部接続を行うインターフェイス"""

    @abstractmethod
    def load_meta(self, id : ItemId) -> np.ndarray:
        """外部からItemIdに対応したメタ情報を取得する"""

        pass

    @abstractmethod
    def load_model(self, model_name : tuple[str, str], device : str) -> Any:
        """外部からmodel_nameに対応したモデルを読み込む"""

        pass

    @abstractmethod
    def load_Index_file(self) -> Any :
        """外部からmodel_nameに対応したインデックスを読み込む"""

        pass