from functools import cache
import re
import traceback
import numpy as np
import faiss
import tqdm

from app.application.accessor import Accessor
from app.domain.domain_object import (
    ImageItem,
    ImageId,
    Image,
    ImageName,
    Model,
    ResultImageItemList,
    Score,
    Tokenizer,
    UploadImage,
    ModelItem,
    ModelId,
    ResultImageItem,
    UploadText,
)
from app.domain.repository import Repository


class Usecase:
    """このアプリケーションの動作を実装するクラス"""

    def __init__(self, repository: Repository, accessor: Accessor) -> None:
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

    def format_search_query(self, search_query_obj) -> str:
        return np.array2string(search_query_obj, separator=', ', suppress_small=True)

    def parse_search_query(self, search_query_text: str):
        try:
            return np.fromstring(search_query_text.strip('[]'), sep=',', dtype=np.float32).reshape(1, -1)
        except ValueError:
            print("Invalid input format for a numpy array.")
            return None

    # ===================================================================

    def similarity_eval(
        self, item_list: list[ImageItem], index, query_features, result_size=2048
    ) -> list[ResultImageItem]:
        print(f'shape:{query_features.shape}')
        # 正規化
        faiss.normalize_L2(query_features)
        item_list_length: int = len(item_list)
        if result_size > item_list_length:
            result_size = item_list_length

        index.nprobe = 256

        print(f"index len:{index.ntotal}")
        print(f"averaged_query_features:{query_features}")
        distances, indices = index.search(query_features, k=result_size)

        sorted_items: list[ImageItem] = sorted(
            item_list, key=lambda x: x.display_name.name
        )

        item_distances: dict[int, float] = {}

        for matched_indices, matched_distances in zip(indices, distances):
            for item_id, item_distance in zip(matched_indices, matched_distances):
                if item_id in item_distances:
                    item_distances[item_id] += item_distance
                else:
                    item_distances[item_id] = item_distance

        result_image_items: list[ResultImageItem] = []
        for item_id, total_distance in item_distances.items():
            try:
                result_image_items.append(
                    ResultImageItem(sorted_items[item_id], Score(total_distance))
                )
            except IndexError:
                traceback.print_exc()
                continue

        print(f"results len:{len(result_image_items)}")
        return result_image_items

    def aesthetic_quality_eval(self, model_id, scores, aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max) -> list[ResultImageItem]:
        if aesthetic_quality_beta == 0 and aesthetic_quality_range_min <= 0 and aesthetic_quality_range_max >= 10:
            return scores
        aesthetic_quality_item: dict[
            ImageId, float
        ] = self._accessor.load_aesthetic_quality_list(model_id)
        new_scores = []
        for i in scores:
            try:
                aesthetic_quality_score: float = aesthetic_quality_item[i.item.id]
                if aesthetic_quality_score >= aesthetic_quality_range_min and aesthetic_quality_score <= aesthetic_quality_range_max:
                    new_score = i.score.score * (1-aesthetic_quality_beta**2) + aesthetic_quality_score * aesthetic_quality_beta
                else:
                    new_score = 0
                new_scores.append(ResultImageItem(i.item, Score(new_score)))
            except IndexError:
                traceback.print_exc()
                continue

        return new_scores

    def search_text(self, model_id: ModelId, text: UploadText, aesthetic_quality_beta: float, aesthetic_quality_range_min: float, aesthetic_quality_range_max: float) -> ResultImageItemList:
        """文字列から検索する"""

        # モデルの読み込み
        model: Model = self._accessor.load_model(model_id)

        # テキストの埋め込みを計算
        tokenizer: Tokenizer = self._accessor.load_tokenizer(model_id)
        features = (
            model.model_obj[0]
            .encode_text(tokenizer.tokenizer_obj([text.text]))
            .to("cpu")
            .detach()
            .numpy()
            .copy()
        )
        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=self._accessor.load_index_item_list(model_id),
            index=index,
            query_features=features,
        )

        scores = self.aesthetic_quality_eval(model_id, scores, aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max)
        return ResultImageItemList(scores, self.format_search_query(features.copy()))

    def search_image(
        self, model_id: ModelId, id_list: list[ImageId], aesthetic_quality_beta :float, aesthetic_quality_range_min: float, aesthetic_quality_range_max: float
    ) -> ResultImageItemList:
        """画像から検索する"""

        # モデルの読み込み
        model_obj: Model = self._accessor.load_model(model_id)
        model, _, preprocess = (
            model_obj.model_obj[0],
            model_obj.model_obj[1],
            model_obj.model_obj[2],
        )

        # 選択した画像のmetaを結合する
        temp = []
        for select_image_id in id_list:
            # テキストの埋め込みを計算
            load_image = self._repository.load_image(select_image_id).to_ptl_image()
            image = preprocess(load_image).unsqueeze(0).to("cpu")
            meta = model.encode_image(image).to("cpu").detach().numpy().copy()
            temp.append(meta)

        # クエリベクトルの平均を計算
        features: np.ndarray = np.vstack(temp).mean(axis=0).reshape(1, -1)

        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=self._accessor.load_index_item_list(model_id),
            index=index,
            query_features=features,
        )

        scores = self.aesthetic_quality_eval(model_id, scores, aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max)
        return ResultImageItemList(scores, self.format_search_query(features))

    def search_name(self, model_id: ModelId, text: UploadText, is_regexp: bool, aesthetic_quality_beta: float, aesthetic_quality_range_min: float, aesthetic_quality_range_max: float) -> ResultImageItemList:
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

        scores = self.aesthetic_quality_eval(model_id, scores, aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max)
        return ResultImageItemList(scores, "")

    def search_upload_image(
        self, model_id: ModelId, image: UploadImage
    ) -> ResultImageItemList:
        """アップロードされた画像から検索する"""

        return ResultImageItemList([
            ResultImageItem(
                item=ImageItem(id=ImageId(id=""), display_name=ImageName(name="")),
                score=Score(score=0.0),
            )
        ], "")

    def search_random(
        self, model_id: ModelId, aesthetic_quality_beta: float, aesthetic_quality_range_min: float, aesthetic_quality_range_max: float
    ) -> ResultImageItemList:
        """乱数から検索する"""

        features: np.ndarray = np.random.normal(0, 1, [1, 768]).astype(np.float32)

        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=self._accessor.load_index_item_list(model_id),
            index=index,
            query_features=features,
        )
        return ResultImageItemList(scores, self.format_search_query(features))

    def search_query(
        self, model_id: ModelId, search_query: str, aesthetic_quality_beta: float, aesthetic_quality_range_min: float, aesthetic_quality_range_max: float
    ) -> ResultImageItemList:
        """クエリから検索する"""

        features: np.ndarray = self.parse_search_query(search_query)

        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=self._accessor.load_index_item_list(model_id),
            index=index,
            query_features=features,
        )

        scores: list[ResultImageItem] = self.aesthetic_quality_eval(model_id, scores, aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max)
        return ResultImageItemList(scores, self.format_search_query(features))

    def add_text_features(
        self, model_id: ModelId, text: UploadText, search_query: str, strength: float, aesthetic_quality_beta: float, aesthetic_quality_range_min: float, aesthetic_quality_range_max: float
    ) -> ResultImageItemList:
        """クエリにstrengthの強さ分テキストの特徴を足してから検索する"""

        # モデルの読み込み
        model: Model = self._accessor.load_model(model_id)

        # テキストの埋め込みを計算
        tokenizer: Tokenizer = self._accessor.load_tokenizer(model_id)

        query_features: np.ndarray = self.parse_search_query(search_query)

        text_features = (
            model.model_obj[0]
            .encode_text(tokenizer.tokenizer_obj([text.text]))
            .to("cpu")
            .detach()
            .numpy()
            .copy()
        )

        faiss.normalize_L2(text_features)
        features = query_features + text_features * strength

        # indexを読み込み
        index = self._accessor.load_index_file(model_id)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=self._accessor.load_index_item_list(model_id),
            index=index,
            query_features=features,
        )

        scores: list[ResultImageItem] = self.aesthetic_quality_eval(model_id, scores, aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max)
        return ResultImageItemList(scores, self.format_search_query(features))

# ===================================================================
