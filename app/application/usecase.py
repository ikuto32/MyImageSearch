




from app.domain.domain_object import Item, ItemId, Image, ItemName, SearchResult
from app.domain.repository import Repository


class Usecase:
    """このアプリケーションの動作を実装するクラス"""

    def __init__(self, repository : Repository):

        self._repository = repository
        self._id_to_items: dict[ItemId, Item] = {}


    def load_items(self) -> None:
        """画像の読み込み"""

        #Itemの読み込み
        items = self._repository.load_items()

        #IDとItemの対応を作成
        self._id_to_items = dict(map(lambda i : (i.get_id(), i), items))

 
    def get_items(self) -> list[Item]:
        """すべてのItemを取得する"""

        return self._id_to_items.values()
    
    def get_ids(self) -> list[ItemId]:
        """すべてのIDを取得する"""

        return self._id_to_items.keys()

    def get_image(self, id : str) -> Image:
        """IDから画像を取得する"""

        id : ItemId = ItemId(id)
        return self._repository.load_image_bytes(id)
    
    def get_item_name(self, id : str) -> ItemName:
        """項目の名前を取得する"""

        id : ItemId = ItemId(id)
        return self._id_to_items.get(id).get_name()
    

    def search_from_text(self, text : str) -> SearchResult:
        """文字列から検索する"""

        # モデルの読み込み
        model, _, _ = util.loadModel(itemlist.model, device="cpu")

        # テキストの埋め込みを計算
        features = util.encode_text(model, text)

        # indexを読み込み
        index = util.loadIndexFile(meta_dir)

        # 類似度を計算する
        scores = util.eval(meta_files, index, features)
        return 
    

    def search_from_image(self, id : list[str]) -> SearchResult:
        """画像から検索する"""

        return
    

    def search_from_upload_image(self, base64 : str) -> SearchResult:
        """アップロードされた画像から検索する"""

        return