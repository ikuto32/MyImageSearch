
import json


class ItemId:
    "項目を一意に識別する値クラス"

    def __init__(self, id: str):
        
        self._id = id

    def get_id(self) -> str:
        return self._id
    
    def __str__(self) -> str:
        return self._id
    
class ItemName:
    "項目の名前を示す値クラス"

    def __init__(self, name: str):

        self._name = name

    def get_name(self) -> str:
        return self._name
    


class Item:
    "各項目を表すエンティティクラス"



    def __init__(self, id : ItemId, name : ItemName):

        self._id : ItemId = id 
        self._name : ItemName = name

    def get_id(self) -> ItemId:
        return self._id
    
    def get_name(self) -> ItemName:
        return self._name



class Image:
    "画像を表すエンティティクラス"

    def __init__(self, binary : bytes, mime_type : str):
        
        self._binary = binary
        self._mime_type = mime_type

    def get_binary(self) -> bytes:

        return self._binary
    
    def get_mime_type(self) -> str:

        return self._mime_type
    










class Score:
    "スコアを示す値クラス"

    def __init__(self, score: float):
        
        self._score = score

    def getScore(self) -> float:
        return self._score
    

class SearchResultItem:
    "検索結果の一つの項目を示すエンティティクラス"

    def __init__(self, id : ItemId, score : Score):

        self._id = id
        self._score = score

    def to_json(self) -> str:

        return json.dumps({
            
            "id": self._id.getId(),
            "score": self._score.getScore()
        })


class SearchResult:
    "検索結果を示すエンティティクラス"

    def __init__(self, items : list[SearchResultItem]):

        self._items = items

    def to_json(self) -> str:

        #TODO 未実装 JSON
        return 