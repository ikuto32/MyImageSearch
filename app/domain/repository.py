

from abc import abstractmethod
from abc import ABCMeta

from app.domain.domain_object import Image, ItemId, Item

class Repository(metaclass=ABCMeta):
    """ドメイン(アプリケーションの役割)として必要な外部接続を行うインターフェイス"""

    @abstractmethod
    def load_items(self) -> list[Item]:
        """外部からItemのリストを取得する"""
    
        pass

    @abstractmethod
    def load_image_bytes(self, id : ItemId) -> Image:
        """外部からItemIdに対応した画像を取得する"""
        
        pass





    