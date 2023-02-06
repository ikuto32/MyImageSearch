
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

    def eval(self, item_list : list[ImageItem], index, features, k=2048):
        
        faiss.normalize_L2(features)
        if k > len(item_list):
            k = len(item_list)
        
        index.nprobe = 64

        print(index.ntotal)
        D, I = index.search(features, k=k)
        
        temp = [0 for _ in item_list]
        id_list = sorted(item_list, key=lambda x:x.display_name.name)
        for i, d in zip(I, D):
            for id, distance in zip(i, d):
                temp[id] += distance
        scores = []
        for i, score in enumerate(temp):
            scores.append([id_list[i].id.id, score])
        return scores

    def search_text(self, model_id : ModelId, text : UploadText) -> list[ResultImageItem]:
        """文字列から検索する"""

        # モデルの読み込み
        model = self._accessor.load_model(model_id)

        # テキストの埋め込みを計算
        tokenizer = self._accessor.load_tokenizer(model_id)
        features = model.model_obj[0].encode_text(tokenizer([text.text])).to("cpu").detach().numpy().copy()

        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores = self.eval(item_list=self.get_all_image_item(), index=index, features=features)
        return scores
    

    def search_image(self, model_id : ModelId, id_list : list[ImageId]) -> list[ResultImageItem]:
        """画像から検索する"""

        if not id_list:
            return
        
        # モデルの読み込み
        model_obj = self._accessor.load_model(model_id)
        model, _, preprocess = model_obj.model_obj[0], model_obj.model_obj[1], model_obj.model_obj[2]
        
        # 選択した画像のmetaを結合する
        temp = []
        for select_image_id in id_list:
            # テキストの埋め込みを計算
            load_image = self._repository.load_image(select_image_id).to_ptl_image()
            image = preprocess(load_image).unsqueeze(0).to("cpu")
            meta = model.encode_image(image).to("cpu").detach().numpy().copy()
            temp.append(meta)

        features = np.vstack(temp)

        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores = self.eval(item_list=self.get_all_image_item(), index=index, features=features)
        return scores
    

    def search_upload_image(self, model_id : ModelId, image : UploadImage) -> list[ResultImageItem]:
        """アップロードされた画像から検索する"""

        return


#===================================================================

