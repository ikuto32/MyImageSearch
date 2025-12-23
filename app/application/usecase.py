import logging
import re
import traceback
from typing import Any
import numpy as np
import faiss
import tqdm

from app.application.accessor import Accessor
from app.domain.domain_object import (
    ImageItem,
    ImageId,
    Image,
    ImageName,
    ImageTags,
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
        self._logger = logging.getLogger(__name__)
        self._repository: Repository = repository
        self._accessor: Accessor = accessor
        self._id_to_image_items: dict[ImageId, ImageItem] = {}
        self._image_items: list[ImageItem] = []

        # 画像項目の読み込み
        self._image_items = self._repository.load_all_image_item()

        # 画像IDと画像項目の対応を作成
        self._id_to_image_items = dict(map(lambda i: (i.id, i), self._image_items))

    def _get_index_and_items(
        self, model_id: ModelId, aesthetic_model_name: str
    ) -> tuple[Any, list[ImageItem]]:
        """指定モデル用のインデックスとメタ情報付き画像一覧を取得する"""

        return self._accessor.load_index_with_metadata(model_id, aesthetic_model_name)

    # ===================================================================

    def get_all_image_item(self) -> list[ImageItem]:
        """すべての画像項目を取得する"""

        return list(self._image_items)

    def get_image_items_by_page(self, page: int, size: int) -> list[ImageItem]:
        """ページングして画像項目を取得する"""

        if size <= 0:
            return []

        normalized_page = max(page, 0)
        start_index = normalized_page * size
        end_index = start_index + size

        return self._image_items[start_index:end_index]

    def get_image_item(self, id: ImageId) -> ImageItem:
        """画像IDから画像項目を取得する"""

        return self._id_to_image_items[id]

    def get_image(self, id: ImageId) -> Image:
        """画像IDから画像項目の画像を取得する"""

        return self._repository.load_image(id)

    def get_small_image(self, id: ImageId) -> Image:
        """画像IDから縮小画像を取得する"""

        return self._repository.load_small_image(id)

    def get_image_metadata(self, model_id: ModelId, image_id: ImageId) -> dict[str, str | float]:
        """画像のタグや評価などのメタデータを取得する"""

        _, items = self._get_index_and_items(model_id, "original")
        for item in items:
            if item.id == image_id:
                return {
                    "tags": item.tags.tags,
                    "style_cluster": item.style_cluster,
                    "rating": item.rating,
                    "aesthetic_quality": item.aesthetic_quality or 0.0,
                }
        return {
            "tags": "",
            "style_cluster": "",
            "rating": "",
            "aesthetic_quality": 0.0,
        }

    def get_rating_list(self, model_id: ModelId) -> dict[ImageId, str]:
        """画像のrating一覧を取得する"""

        _, items = self._get_index_and_items(model_id, "original")
        return {item.id: item.rating for item in items}

    # ===================================================================

    def get_all_model(self) -> list[ModelItem]:
        """すべての検索モデルを取得する"""

        return self._repository.load_all_model_item()

    # ===================================================================

    def format_search_query(self, search_query_obj) -> str:
        """検索クエリのnumpy配列を文字列表現に整形する。

        Args:
            search_query_obj (np.ndarray): 検索クエリとして扱う特徴量配列。1行ベクトルを想定。

        Returns:
            str: `np.array2string`でフォーマットした文字列表現。返却時に精度が抑制される。
        """
        return np.array2string(search_query_obj, separator=", ", suppress_small=True)

    def parse_search_query(self, search_query_text: str):
        """文字列表現の検索クエリをnumpy配列に変換する。

        Args:
            search_query_text (str): "[0.1, 0.2, ...]"形式を想定したテキスト。

        Returns:
            np.ndarray | None: `float32`の1行ベクトルに整形した配列。フォーマット不正時は`None`を返却し、エラー内容を標準出力へ記録する。
        """
        try:
            return np.fromstring(
                search_query_text.strip("[]"), sep=",", dtype=np.float32
            ).reshape(1, -1)
        except ValueError:
            self._logger.error("Invalid input format for a numpy array: %s", search_query_text)
            return None

    # ===================================================================

    def similarity_eval(
        self, item_list: list[ImageItem], index, query_features, result_size=2048
    ) -> list[ResultImageItem]:
        """ベクトル類似度を計算し、結果を`ResultImageItem`一覧として返す。

        Args:
            item_list (list[ImageItem]): インデックスと同順でソートされる画像メタ情報一覧。
            index (Any): FAISSインデックス。`nprobe`を64に更新して探索する。
            query_features (np.ndarray): `(1, 次元数)`の特徴量配列。検索前にL2正規化を実施する副作用がある。
            result_size (int, optional): 返却件数の上限。デフォルトは2048で、アイテム数を超える場合は自動で短縮される。

        Returns:
            list[ResultImageItem]: 距離に基づき生成した結果リスト。インデックス外参照が発生したIDはスキップし、`traceback`を標準出力へ表示する。
        """
        self._logger.info("query_features shape: %s", query_features.shape)
        # 正規化
        faiss.normalize_L2(query_features)
        item_list_length: int = len(item_list)
        if result_size > item_list_length:
            result_size = item_list_length

        index.nprobe = 64

        self._logger.info("index size: %s", index.ntotal)
        self._logger.info("averaged_query_features: %s", query_features)
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

        self._logger.info("results length: %s", len(result_image_items))
        return result_image_items

    def apply_aesthetic_quality_filter(
        self,
        model_id,
        scores,
        aesthetic_quality_beta,
        aesthetic_quality_range_min,
        aesthetic_quality_range_max,
        aesthetic_model_name,
    ) -> list[ResultImageItem]:
        """類似度スコアに審美性評価を組み合わせて再スコアリングする。

        Args:
            model_id (ModelId): 対象モデルID。
            scores (list[ResultImageItem]): 類似度計算済みの結果リスト。
            aesthetic_quality_beta (float): 審美性評価を加重する係数。0で無効化。
            aesthetic_quality_range_min (float): 許容下限スコア。
            aesthetic_quality_range_max (float): 許容上限スコア。
            aesthetic_model_name (str): 審美性モデル名（閾値判定の文脈情報としてのみ使用）。

        Returns:
            list[ResultImageItem]: 審美性スコアが`None`のアイテムや範囲外のアイテムはスキップまたは0スコアとして扱った再計算結果。
        """
        if (
            aesthetic_quality_beta == 0
            and aesthetic_quality_range_min <= 0
            and aesthetic_quality_range_max >= 10
        ):
            return scores
        new_scores = []
        for i in scores:
            aesthetic_quality_score = i.item.aesthetic_quality
            if aesthetic_quality_score is None:
                continue
            if aesthetic_quality_range_min <= aesthetic_quality_score <= aesthetic_quality_range_max:
                new_score = (
                    i.score.score * (1 - aesthetic_quality_beta**2)
                    + aesthetic_quality_score * aesthetic_quality_beta
                )
            else:
                new_score = 0
            new_scores.append(ResultImageItem(i.item, Score(new_score)))

        return new_scores

    def search_text(
        self,
        model_id: ModelId,
        text: UploadText,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
    ) -> ResultImageItemList:
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
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            query_features=features,
        )

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return ResultImageItemList(scores, self.format_search_query(features.copy()))

    def search_image(
        self,
        model_id: ModelId,
        id_list: list[ImageId],
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
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
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            query_features=features,
        )

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return ResultImageItemList(scores, self.format_search_query(features))

    def search_name(
        self,
        model_id: ModelId,
        text: UploadText,
        is_regexp: bool,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
    ) -> ResultImageItemList:
        """文字列から名前検索する"""

        scores: list[ResultImageItem] = []
        # 類似度を計算する
        _, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        for image_item in tqdm.tqdm(item_list):
            name: str = image_item.display_name.name
            # print(args.get("trueRegexp"))
            if is_regexp:
                has_match: bool = re.search(text.text, name) is not None
            else:
                has_match = text.text in name

            if has_match:
                scores.append(ResultImageItem(image_item, Score(1.0)))

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return ResultImageItemList(scores, "")

    def search_upload_image(
        self, model_id: ModelId, image: UploadImage
    ) -> ResultImageItemList:
        """アップロードされた画像から検索する"""

        # モデルの読み込み
        model_obj: Model = self._accessor.load_model(model_id)
        model, _, preprocess = (
            model_obj.model_obj[0],
            model_obj.model_obj[1],
            model_obj.model_obj[2],
        )

        # アップロードされた画像を前処理
        load_image = Image(binary=image.binary, content_type=image.content_type).to_ptl_image()
        image_tensor = preprocess(load_image).unsqueeze(0).to("cpu")
        features = model.encode_image(image_tensor).to("cpu").detach().numpy().copy()

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, "original")

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            query_features=features,
        )

        return ResultImageItemList(scores, self.format_search_query(features))

    def search_random(
        self,
        model_id: ModelId,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
    ) -> ResultImageItemList:
        """乱数から検索する"""

        features: np.ndarray = np.random.normal(0, 1, [1, 768]).astype(np.float32)

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            query_features=features,
        )
        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return ResultImageItemList(scores, self.format_search_query(features))

    def search_query(
        self,
        model_id: ModelId,
        search_query: str,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
    ) -> ResultImageItemList:
        """クエリから検索する"""

        features = self.parse_search_query(search_query)

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            query_features=features,
        )

        scores: list[ResultImageItem] = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return ResultImageItemList(scores, self.format_search_query(features))

    def add_text_features(
        self,
        model_id: ModelId,
        text: UploadText,
        search_query: str,
        strength: float,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
    ) -> ResultImageItemList:
        """クエリにstrengthの強さ分テキストの特徴を足してから検索する"""

        # モデルの読み込み
        model: Model = self._accessor.load_model(model_id)

        # テキストの埋め込みを計算
        tokenizer: Tokenizer = self._accessor.load_tokenizer(model_id)

        query_features = self.parse_search_query(search_query)

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
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            query_features=features,
        )

        scores: list[ResultImageItem] = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name
        )
        return ResultImageItemList(scores, self.format_search_query(features))


    def search_tags(
        self,
        model_id: ModelId,
        text: UploadText,
        is_regexp: bool,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
    ) -> ResultImageItemList:
        """文字列からタグ検索する"""

        scores: list[ResultImageItem] = []
        # 類似度を計算する
        _, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        for image_item in tqdm.tqdm(item_list):
            tags: str = image_item.tags.tags
            # print(args.get("trueRegexp"))
            if is_regexp:
                has_match: bool = re.search(text.text, tags) is not None
            else:
                has_match = text.text in tags

            if has_match:
                scores.append(ResultImageItem(image_item, Score(1.0)))

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return ResultImageItemList(scores, "")

    def search_style_cluster(
        self,
        model_id: ModelId,
        text: UploadText,
        is_regexp: bool,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
    ) -> ResultImageItemList:
        """style_cluster を文字列検索する"""

        scores: list[ResultImageItem] = []
        _, item_list = self._get_index_and_items(model_id, aesthetic_model_name)

        for image_item in tqdm.tqdm(item_list):
            style_cluster = image_item.style_cluster

            if is_regexp:
                has_match: bool = re.search(text.text, style_cluster) is not None
            else:
                has_match = text.text in style_cluster

            if has_match:
                scores.append(ResultImageItem(image_item, Score(1.0)))

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return ResultImageItemList(scores, "")

    def get_images_zip(self, id_list):
        """指定された画像IDからZIPバッファを生成する。

        Args:
            id_list (Iterable[str]): 画像IDの反復可能オブジェクト。各IDは`ImageId`へ変換される。

        Returns:
            BytesIO: 取得できた画像とファイル名のペアを元に生成したZIPデータ。無効IDや読み込み失敗時は標準出力に理由を記録し、該当IDをスキップする。
        """
        self._logger.info("start:get_images_zip")
        images_with_names = []
        for image_id_str in tqdm.tqdm(id_list):
            image_id = ImageId(image_id_str)  # ImageId型に変換
            try:
                image_with_name: tuple[Image, ImageName] = self._repository.load_image(image_id), self._repository.get_image_name(image_id)
                images_with_names.append(image_with_name)
            except (ValueError, FileNotFoundError) as e:
                self._logger.warning("画像IDが無効のためスキップ: %s", e)

        zip_buffer = self._repository.create_zip_from_images(images_with_names)
        return zip_buffer
# ===================================================================
