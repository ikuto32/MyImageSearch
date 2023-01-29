
from dataclasses import dataclass
from typing import Any



#=====================================================
# 画像 関連

@dataclass(frozen=True)
class ImageId:

    id : str

    def __hash__(self):
        return hash(self.id)

  
@dataclass(frozen=True)
class ImageName:

    name : str


@dataclass(frozen=True)
class Image:

    binary : bytes
    content_type : str


@dataclass(frozen=True)
class ImageItem:

    id : ImageId
    display_name : ImageName


#=====================================================
# 検索モデル 関連

@dataclass(frozen=True)
class ModelId:

    id : str


@dataclass(frozen=True)
class ModelName:

    name : str


@dataclass(frozen=True)
class ModelItem:

    id : ModelId
    display_name : ModelName


@dataclass(frozen=True)
class Model:

    model_obj: Any


#=====================================================
# 検索結果 関連


@dataclass(frozen=True)
class Score:
    """スコアを示す値クラス"""

    score : float
    

@dataclass(frozen=True)
class ResultImageItem:
    """検索結果の画像項目を表すエンティティクラス"""

    item : ImageItem
    score : Score


#=====================================================
# 検索パラメータ 関連



@dataclass(frozen=True)
class UploadText:
    "(入力された)検索文字列を示す値クラス"

    text : str


@dataclass(frozen=True)
class UploadImage:
    """(入力された)検索画像を示す値クラス"""

    binary : bytes
    content_type : str

