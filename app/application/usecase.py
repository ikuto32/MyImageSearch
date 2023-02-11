
import re
import numpy as np
import faiss
import tqdm

from app.application.accessor import Accessor
from app.domain.domain_object import ImageItem, ImageId, Image, ImageName, Model, Score, Tokenizer, UploadImage, ModelItem, ModelId, ResultImageItem, UploadText
from app.domain.repository import Repository


class Usecase:
    """このアプリケーションの動作を実装するクラス"""

    def __init__(
        self,
        repository: Repository,
        accessor: Accessor
    ) -> None:

        self._repository: Repository = repository
        self._accessor: Accessor = accessor
        self._id_to_image_items: dict[ImageId, ImageItem] = {}

        # 画像項目の読み込み
        items: list[ImageItem] = self._repository.load_all_image_item()

        # 画像IDと画像項目の対応を作成
        self._id_to_image_items = dict(map(lambda i: (i.id, i), items))


# ===================================================================


    def get_all_image_item(self) -> list[ImageItem]:
        """すべての画像項目を取得する"""

        return list(self._id_to_image_items.values())

    def get_image_item(self, id: ImageId) -> ImageItem:
        """画像IDから画像項目を取得する"""

        return self._id_to_image_items[id]

    def get_image(self, id: ImageId) -> Image:
        """画像IDから画像項目の画像を取得する"""

        return self._repository.load_image(id)


# ===================================================================


    def get_all_model(self) -> list[ModelItem]:
        """すべての検索モデルを取得する"""

        return self._repository.load_all_model_item()


# ===================================================================

    

    def eval(self, item_list: list[ImageItem], index, features, result_size=2048) -> list[ResultImageItem]:

        faiss.normalize_L2(features)
        if result_size > len(item_list):
            result_size = len(item_list)

        index.nprobe = 64

        print(index.ntotal)
        D, I = index.search(features, k=result_size)

        temp: list[int] = [0 for _ in item_list]
        sorted_item_list: list[ImageItem] = sorted(
            item_list, key=lambda x: x.display_name.name)
        for i, d in zip(I, D):
            for id, distance in zip(i, d):
                temp[id] += distance
        results: list[ResultImageItem] = []
        for i, score in enumerate(temp):
            results.append(ResultImageItem(sorted_item_list[i], Score(score)))

        return results

    def search_text(self, model_id: ModelId, text: UploadText) -> list[ResultImageItem]:
        """文字列から検索する"""

        # モデルの読み込み
        model: Model = self._accessor.load_model(model_id)

        # テキストの埋め込みを計算
        tokenizer: Tokenizer = self._accessor.load_tokenizer(model_id)
        features = model.model_obj[0].encode_text(
            tokenizer.tokenizer_obj([text.text])).to("cpu").detach().numpy().copy()
        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.eval(item_list=self._accessor.load_index_item_list(),
                                                   index=index, features=features)
        
        aesthetic_quality_item: dict[ImageId, float] = self._accessor.load_aesthetic_quality_list()
        scores = list(map(lambda i: ResultImageItem(i.item, Score(i.score.score * aesthetic_quality_item[i.item.id])), scores))

        return scores

    def search_image(self, model_id: ModelId, id_list: list[ImageId]) -> list[ResultImageItem]:
        """画像から検索する"""

        
        # モデルの読み込み
        model_obj: Model = self._accessor.load_model(model_id)
        model, _, preprocess = model_obj.model_obj[0], model_obj.model_obj[1], model_obj.model_obj[2]

        # 選択した画像のmetaを結合する
        temp = []
        for select_image_id in id_list:
            # テキストの埋め込みを計算
            load_image = self._repository.load_image(
                select_image_id).to_ptl_image()
            image = preprocess(load_image).unsqueeze(0).to("cpu")
            meta = model.encode_image(image).to("cpu").detach().numpy().copy()
            temp.append(meta)

        features: np.ndarray = np.vstack(temp)

        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.eval(item_list=self._accessor.load_index_item_list(),
                                                   index=index, features=features)
        
        aesthetic_quality_item: dict[ImageId, float] = self._accessor.load_aesthetic_quality_list()
        scores = list(map(lambda i: ResultImageItem(i.item, Score(i.score.score * aesthetic_quality_item[i.item.id])), scores))
        return scores

    def search_name(self, text: UploadText, is_regexp: bool) -> list[ResultImageItem]:
        """文字列から名前検索する"""

        scores: list[ResultImageItem] = []
        # 類似度を計算する
        for image_item in tqdm.tqdm(self._repository.load_all_image_item()):
            name: str = image_item.display_name.name
            # print(args.get("trueRegexp"))
            if is_regexp:
                hasMatch: bool = re.search(text.text, name) != None
            else:
                hasMatch = text.text in name

            if hasMatch:
                scores.append(ResultImageItem(image_item, Score(1.0)))
            else:
                scores.append(ResultImageItem(image_item, Score(0.0)))
        aesthetic_quality_item: dict[ImageId, float] = self._accessor.load_aesthetic_quality_list()
        scores = list(map(lambda i: ResultImageItem(i.item, Score(i.score.score * aesthetic_quality_item[i.item.id])), scores))
        return scores

    def search_upload_image(self, model_id: ModelId, image: UploadImage) -> list[ResultImageItem]:
        """アップロードされた画像から検索する"""

        return [ResultImageItem(item=ImageItem(id=ImageId(id=""), display_name=ImageName(name="")), score=Score(score=0.0))]


# ===================================================================
