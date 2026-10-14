from dataclasses import dataclass
from io import BytesIO
from typing import NewType

from PIL import Image as PILImage

from app.domain import limits
from app.domain.errors import ResourceLimitError

# =====================================================
# 画像 関連

# 画像を一意に識別するID。
ImageId = NewType("ImageId", str)


@dataclass(frozen=True, slots=True)
class Image:
    """画像バイナリとコンテンツタイプを保持し、Pillow画像へ変換できる。"""

    binary: bytes
    content_type: str
    # 画像ルートからの相対パス。アップロードされた画像では空。
    name: str = ""

    def to_pil_image(self) -> PILImage.Image:
        """バイナリがPillowで解釈可能な形式である前提で画像に変換する。"""
        if len(self.binary) > limits.MAX_IMAGE_BYTES:
            raise ResourceLimitError("検索に使う画像は64MiB以内で指定してください。")
        try:
            with PILImage.open(BytesIO(self.binary)) as image:
                if image.width * image.height > limits.MAX_IMAGE_PIXELS:
                    raise ResourceLimitError("検索に使う画像は6400万画素以内で指定してください。")
                # The index builder encodes the stored pixel orientation. Keep
                # that convention for search; thumbnails may orient for display.
                return image.copy()
        except PILImage.DecompressionBombError:
            raise ResourceLimitError("画像の画素数が大きすぎます。縮小して再試行してください。") from None


@dataclass(frozen=True, slots=True)
class ImageItem:
    """画像メタデータ一式。評価値やクラスタ情報は未設定も許容。"""

    id: ImageId
    # 画像ルートからの相対パス。表示名を兼ねる。
    name: str
    tags: str = ""
    rating: str = ""
    style_cluster: str = ""
    aesthetic_quality: float | None = None


# =====================================================
# 検索モデル 関連

@dataclass(frozen=True, slots=True)
class ModelId:
    """モデル名と事前学習設定の組み合わせで一意となるID。"""

    model_name: str
    pretrained: str


@dataclass(frozen=True, slots=True)
class ModelItem:
    """検索に利用可能なモデルと、ユーザー向けの表示名。"""

    id: ModelId
    display_name: str


# =====================================================
# 検索結果 関連

@dataclass(frozen=True, slots=True)
class SearchResult:
    """検索結果の画像項目とそのスコア。"""

    item: ImageItem
    score: float


@dataclass(frozen=True, slots=True)
class SearchResults:
    """スコア順の検索結果と、同じ検索を再実行するための保存クエリ。"""

    items: list[SearchResult]
    search_query: str
