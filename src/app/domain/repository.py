

from abc import abstractmethod
from abc import ABCMeta
import numpy as np

from app.domain.domain_object import Image, ItemId, Item


class Repository(metaclass=ABCMeta):


    @abstractmethod
    def load_items(self) -> list[Item]:
        "外部からItemのリストを取得する"
    
        pass

    @abstractmethod
    def load_image_bytes(self, id : ItemId) -> Image:
        "外部からItemIdに対応した画像を取得する"
        
        pass


    @abstractmethod
    def load_meta(self, id : ItemId) -> np.ndarray:
        "外部からItemIdに対応したメタ情報を取得する"

        pass

    

    