from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import BinaryIO

from app.domain.domain_object import Image, ImageId, ImageItem, ModelId, ModelItem
from app.domain.rating import RatingFilter


class Repository(ABC):
    """ドメイン(検索モデル・画像項目・画像)として必要な外部接続を行うインターフェイス"""

    @abstractmethod
    def load_all_model_item(self) -> list[ModelItem]:
        """検索に利用できるモデルを表示名順に返す。"""

    @abstractmethod
    def count_image_items(self, model_id: ModelId, ratings: RatingFilter = None) -> int:
        """モデルに登録された画像項目のうち、区分に一致する件数を返す。"""

    @abstractmethod
    def load_image_items_by_page(
        self, model_id: ModelId, page: int, size: int, ratings: RatingFilter = None,
    ) -> list[ImageItem]:
        """名前順に並べた画像項目の1ページ分を返す。範囲外のページは空。

        一覧の項目はID・名前・ratingだけを持つ。タグなどは ``find_image_item`` で取得する。
        """

    @abstractmethod
    def find_image_item(self, model_id: ModelId, image_id: ImageId) -> ImageItem | None:
        """画像IDに対応する画像項目を返す。登録されていなければNone。"""

    @abstractmethod
    def load_image(self, image_id: ImageId) -> Image:
        """画像IDに対応するオリジナル画像を返す。

        未知の画像IDや読めないファイルは ``ImageUnavailableError`` で表現する。
        """

    @abstractmethod
    def load_small_image(self, image_id: ImageId) -> Image:
        """画像IDに対応する縮小画像を返す。

        未知の画像IDや読めないファイルは ``ImageUnavailableError`` で表現する。
        """

    def remember_image_items(self, items: Iterable[ImageItem]) -> None:
        """検索結果として返す画像項目を知らせる。

        続く画像の取得でIDの解決を省くための手がかりで、何もしなくてもよい。
        """

    @abstractmethod
    def create_zip_from_images(self, images: Iterable[Image]) -> BinaryIO:
        """画像をZIPにまとめ、先頭へseek済みのストリームを返す。呼び出し側がcloseする。"""
