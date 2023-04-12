
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

    

    def eval(self, item_list: list[ImageItem], index, query_features, result_size=2048) -> list[ResultImageItem]:

        faiss.normalize_L2(query_features)
        item_list_length: int = len(item_list)
        if result_size > item_list_length:
            result_size = item_list_length

        index.nprobe = 64

        print(f"index len:{index.ntotal}")
        distances, indices = index.search(query_features, k=result_size)

        sorted_items: list[ImageItem] = sorted(
            item_list, key=lambda x: x.display_name.name)

        item_distances: dict[int, float] = {}

        for matched_indices, matched_distances in zip(indices, distances):
            for item_id, item_distance in zip(matched_indices, matched_distances):
                if item_id in item_distances:
                    item_distances[item_id] += item_distance
                else:
                    item_distances[item_id] = item_distance

        result_image_items: list[ResultImageItem] = []
        for item_id, total_distance in item_distances.items():
                result_image_items.append(ResultImageItem(sorted_items[item_id], Score(total_distance)))

        print(f"results len:{len(result_image_items)}")
        return result_image_items

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
        scores: list[ResultImageItem] = self.eval(item_list=self._accessor.load_index_item_list(model_id),
                                                   index=index, query_features=features)
        
        aesthetic_quality_item: dict[ImageId, float] = self._accessor.load_aesthetic_quality_list(model_id)

        beta: float = 0.05
        scores = list(map(lambda i: ResultImageItem(i.item, Score(i.score.score + aesthetic_quality_item[i.item.id]*beta)), scores))

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
        scores: list[ResultImageItem] = self.eval(item_list=self._accessor.load_index_item_list(model_id),
                                                   index=index, query_features=features)
        
        aesthetic_quality_item: dict[ImageId, float] = self._accessor.load_aesthetic_quality_list(model_id)
        beta: float = 0.05
        scores = list(map(lambda i: ResultImageItem(i.item, Score(i.score.score + aesthetic_quality_item[i.item.id]*beta)), scores))
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
        return scores

    def search_upload_image(self, model_id: ModelId, image: UploadImage) -> list[ResultImageItem]:
        """アップロードされた画像から検索する"""

        return [ResultImageItem(item=ImageItem(id=ImageId(id=""), display_name=ImageName(name="")), score=Score(score=0.0))]


# ===================================================================
