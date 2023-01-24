




from app.domain.domain_object import Item, ItemId, Image, ItemName, SearchResult
from app.domain.repository import Repository


class Usecase:

    def __init__(self, repository : Repository):

        self._repository = repository


    def load_items(self) -> None:
        """画像の読み込み"""

        #Itemの読み込み
        items = self._repository.load_items()

        #IDとItemの対応を作成
        self._id_to_items : dict[ItemId, Item] = dict(map(lambda i : (i.get_id(), i), items))
        pass

    def get_image(self, id : str) -> Image:
        """IDから画像を取得する"""

        id : ItemId = ItemId(id)
        return self._repository.load_image_bytes(id)
    
    def get_item_name(self, id : str) -> ItemName:
        """項目の名前を取得する"""

        id : ItemId = ItemId(id)
        return self._id_to_items.get(id).get_name()
    

    def searchFromText(self, text : str) -> SearchResult:
        """文字列から検索する"""

        return 
    

    def searchFromImage(self, id : list[str]) -> SearchResult:
        """画像から検索する"""

        return
    

    def searchFromUploadImage(self, base64 : str) -> SearchResult:
        """アップロードされた画像から検索する"""

        return