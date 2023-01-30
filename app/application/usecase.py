
import numpy as np
import faiss

from app.application.accessor import Accessor
from app.domain.domain_object import ImageItem, ImageId, Image, UploadImage, ModelItem, ModelId, ResultImageItem, UploadText
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
        self._id_to_image_items: dict[ImageId, ImageItem] = {}

        #画像項目の読み込み
        items = self._repository.load_all_image_item()

        #画像IDと画像項目の対応を作成
        self._id_to_image_items = dict(map(lambda i : (i.id, i), items))


#===================================================================


    def get_all_image_item(self) -> list[ImageItem]:
        """すべての画像項目を取得する"""

        return self._id_to_image_items.values()

    def get_image_item(self, id : ImageId) -> ImageItem:
        """画像IDから画像項目を取得する"""

        return self._id_to_image_items.get(id)

    def get_image(self, id : ImageId) -> Image:
        """画像IDから画像項目の画像を取得する"""

        return self._repository.load_image(id)


#===================================================================


    def get_all_model(self) -> list[ModelItem]:
        """すべての検索モデルを取得する"""
        
        return self._repository.load_all_model_item()

    
#===================================================================

    def eval(id_list, index, features, k=2048):
        features = features.astype(np.float32)
        faiss.normalize_L2(features)
        if k > len(id_list):
            k = len(id_list)
        
        index.nprobe = 64

        print(index.ntotal)
        D, I = index.search(features, k=k)
        
        temp = [0 for i in id_list]
        for i, d in zip(I, D):
            for id, distance in zip(i, d):
                temp[id] += distance
        scores = []
        for i, score in enumerate(temp):
            scores.append([id_list[i], score])
        return scores

    def search_text(self, model_id : ModelId, text : UploadText) -> list[ResultImageItem]:
        """文字列から検索する"""

        # モデルの読み込み
        model, _, _ = self._accessor.load_model(model_id)

        # テキストの埋め込みを計算
        features = model.encode_text(text)

        # indexを読み込み
        index = self._accessor.load_Index_file(model_id)

        # 類似度を計算する
        scores = self.eval(index, features)
        return
    

    def search_image(self, model_id : ModelId, id : list[ImageId]) -> list[ResultImageItem]:
        """画像から検索する"""

        return
    

    def search_upload_image(self, model_id : ModelId, image : UploadImage) -> list[ResultImageItem]:
        """アップロードされた画像から検索する"""

        return


#===================================================================

