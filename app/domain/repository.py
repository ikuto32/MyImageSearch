

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
import numpy as np

from app.domain.domain_object import Image, ItemId, Item


class Repository(metaclass=ABCMeta):


    @abstractmethod
    def load_items(self) -> list[Item]:
        """外部からItemのリストを取得する"""
    
        pass

    @abstractmethod
    def load_image_bytes(self, id : ItemId) -> Image:
        """外部からItemIdに対応した画像を取得する"""
        
        pass


    @abstractmethod
    def load_meta(self, id : ItemId) -> np.ndarray:
        """外部からItemIdに対応したメタ情報を取得する"""

        pass

    @abstractmethod
    def load_model(self, model_name : tuple[str, str], device : str) -> Any:
        """外部からmodel_nameに対応したモデルを読み込む"""

        pass

    abstractmethod
    def load_Index_file(self) -> Any :
        """外部からmodel_nameに対応したインデックスを読み込む"""

        pass


    