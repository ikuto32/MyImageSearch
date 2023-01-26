


from app.application.accessor import Accessor
from app.domain.domain_object import Item, ItemId, Image, ItemName, SearchResult, SearchResultItem, SearchText
from app.domain.repository import Repository


class Usecase:
    """このアプリケーションの動作を実装するクラス"""

    def __init__(
        self, 
        repository : Repository,
        accessor : Accessor
    ):

        self._repository = repository
        self._accessor = accessor
        self._id_to_items: dict[ItemId, Item] = {}


    def load_items(self) -> None:
        """画像の読み込み"""

        #Itemの読み込み
        items = self._repository.load_items()

        #IDとItemの対応を作成
        self._id_to_items = dict(map(lambda i : (i.id, i), items))


    def get_image(self, id : ItemId) -> Image:
        """IDから画像を取得する"""

        return self._repository.load_image_bytes(id)
    
    def get_item_name(self, id : ItemId) -> ItemName:
        """項目の名前を取得する"""

        return self._id_to_items.get(id).name

    def search_all(self) -> SearchResult:
        """すべてのItemを取得する"""

        items : list[Item] = self._id_to_items.values()
        result_items : list[SearchResultItem] = map(SearchResultItem.create_from_item, items)
        return SearchResult(result_items)

    def search_from_text(self, text : SearchText) -> SearchResult:
        """文字列から検索する"""

        # # モデルの読み込み
        # model, _, _ = util.loadModel(itemlist.model, device="cpu")

        # # テキストの埋め込みを計算
        # features = util.encode_text(model, text)

        # # indexを読み込み
        # index = util.loadIndexFile(meta_dir)

        # # 類似度を計算する
        # scores = util.eval(meta_files, index, features)
        return 
    

    def search_from_image(self, id : list[ItemId]) -> SearchResult:
        """画像から検索する"""

        return
    

    def search_from_upload_image(self, image : Image) -> SearchResult:
        """アップロードされた画像から検索する"""

        return