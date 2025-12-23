from dataclasses import dataclass
from typing import Any
from PIL import Image as PILImage
from io import BytesIO


# =====================================================
# 画像 関連

@dataclass(frozen=True)
class ImageId:
    """画像を一意に識別するID。"""

    id: str

    def __hash__(self):
        return hash(self.id)


@dataclass(frozen=True)
class ImageName:
    """画像の表示名。"""

    name: str


@dataclass(frozen=True)
class ImageTags:
    """検索や分類に使うタグ文字列。"""

    tags: str


@dataclass(frozen=True)
class Image:
    """画像バイナリとコンテンツタイプを保持し、Pillow画像へ変換できる。"""

    binary: bytes
    content_type: str

    def to_ptl_image(self):
        """バイナリがPillowで解釈可能な形式である前提で画像に変換する。"""
        return PILImage.open(BytesIO(self.binary))


@dataclass(frozen=True)
class ImageItem:
    """画像メタデータ一式。評価値やクラスタ情報は未設定も許容。"""

    id: ImageId
    display_name: ImageName
    tags: ImageTags = ImageTags("")
    aesthetic_quality: float | None = None
    rating: str = ""
    style_cluster: str = ""


# =====================================================
# 検索モデル 関連

@dataclass(frozen=True)
class ModelId:
    """モデル名と事前学習設定の組み合わせで一意となるID。"""

    model_name: str
    pretrained: str


@dataclass(frozen=True)
class ModelName:
    """ユーザー向けに表示するモデル名。"""

    name: str


@dataclass(frozen=True)
class ModelItem:
    """検索に利用可能なモデルのメタデータ。"""

    id: ModelId
    display_name: ModelName


@dataclass(frozen=True)
class Model:
    """実際のモデルオブジェクトのラッパー。"""

    model_obj: Any


@dataclass(frozen=True)
class Tokenizer:
    """テキスト前処理用トークナイザオブジェクトのラッパー。"""

    tokenizer_obj: Any


# =====================================================
# 検索結果 関連


@dataclass(frozen=True)
class Score:
    """スコアを示す値クラス"""

    score: float


@dataclass(frozen=True)
class ResultImageItem:
    """検索結果の画像項目を表すエンティティクラス"""

    item: ImageItem
    score: Score


@dataclass(frozen=True)
class ResultImageItemList:
    """検索結果の画像項目のリストを表すエンティティクラス"""

    list: list[ResultImageItem]
    search_query: str


# =====================================================
# 検索パラメータ 関連


@dataclass(frozen=True)
class UploadText:
    """(入力された)検索文字列を示す値クラス"""

    text: str


@dataclass(frozen=True)
class UploadImage:
    """(入力された)検索画像を示す値クラス"""

    binary: bytes
    content_type: str
