
from typing import Iterator


class ItemId:
    """項目を一意に識別する値クラス"""

    def __init__(self, id: str):
        
        self._id = id

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        return self._id == other._id
    
    def __str__(self) -> str:
        return self._id
    
    @property
    def id(self) -> str:
        return self._id
    
class ItemName:
    """項目の名前を示す値クラス"""

    def __init__(self, name: str):

        self._name = name

    
    def __str__(self) -> str:
        return self._name

    @property
    def name(self) -> str:
        return self._name
    


class Item:
    """各項目を表すエンティティクラス"""

    def __init__(self, id : ItemId, name : ItemName):

        self._id : ItemId = id 
        self._name : ItemName = name

    @property
    def id(self) -> ItemId:
        return self._id
    
    @property
    def name(self) -> ItemName:
        return self._name



class Image:
    """画像を表すエンティティクラス"""

    def __init__(self, binary : bytes, mime_type : str):
        
        self._binary = binary
        self._mime_type = mime_type

    @property
    def binary(self) -> bytes:
        return self._binary
    
    @property
    def mime_type(self) -> str:
        return self._mime_type
    




class Score:
    """スコアを示す値クラス"""

    def __init__(self, score: float = 0.0):
        
        self._score = score

    def __str__(self) -> str:
        return self._score

    @property
    def score(self) -> float:
        return self._score
    

class SearchResultItem:
    """検索結果の一つの項目を示すエンティティクラス"""

    @staticmethod
    def create_from_item(item : Item) -> 'SearchResultItem':
        "Itemから検索結果のItemを作成する"

        return SearchResultItem(item.get_id(), Score())


    def __init__(self, id : ItemId, score : Score):

        self._id = id
        self._score = score

    @property
    def id(self) -> ItemId:
        return self._id

    @property
    def score(self) -> Score:
        return self._score



class SearchResult:
    """検索結果を示すエンティティクラス"""

    def __init__(self, items : list[SearchResultItem]):

        self._items = items

    def __iter__(self) -> Iterator[SearchResultItem]:
        return self._items.__iter__()


class SearchText:
    "検索対象の(入力された)文字列"

    def __init__(self, text : str):

        self._text = text
    
    @property
    def text(self):
        return self._text