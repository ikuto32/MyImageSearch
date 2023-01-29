

from abc import abstractmethod
from abc import ABCMeta

from app.domain.domain_object import ImageItem, ImageId, Image, Model, ModelItem, ModelId

class Repository(metaclass=ABCMeta):
    """ドメイン(アプリケーションの役割)として必要な外部接続を行うインターフェイス"""

    @abstractmethod
    def load_all_image_item(self) -> list[ImageItem]:
    
        pass
    
    @abstractmethod
    def load_image(self, id : ImageId) -> Image:
        
        pass


    @abstractmethod
    def load_all_model_item(self) -> list[ModelItem]:

        pass

    @abstractmethod
    def load_model(self, id : ModelId) -> Model:

        pass





    