

from abc import abstractmethod
from abc import ABCMeta
import io

from app.domain.domain_object import ImageItem, ImageId, Image, ImageName, ModelItem

class Repository(metaclass=ABCMeta):
    """ドメイン(アプリケーションの役割)として必要な外部接続を行うインターフェイス"""

    @abstractmethod
    def load_all_image_item(self) -> list[ImageItem]:

        pass

    @abstractmethod
    def load_image(self, image_id: ImageId) -> Image:

        pass

    @abstractmethod
    def load_all_model_item(self) -> list[ModelItem]:

        pass

    @abstractmethod
    def create_zip_from_images(self, images_with_names) -> io.BytesIO:

        pass

    @abstractmethod
    def get_image_name(self, image_id: ImageId) -> ImageName:

        pass