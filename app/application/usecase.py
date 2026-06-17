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
    ResultImageItemList,
    Score,
    UploadImage,
    ModelItem,
    ModelId,
    ResultImageItem,
    UploadText,
)
from app.domain.repository import Repository


class Usecase:
    """このアプリケーションの動作を実装するクラス"""

    def __init__(
        self,
        repository: Repository,
        accessor: Accessor,
        startup_model_id: ModelId,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._repository: Repository = repository
        self._accessor: Accessor = accessor
        self._id_to_image_items: dict[ImageId, ImageItem] = {}
        self._image_items: list[ImageItem] = []

        # 起動時はDBのメタ情報から画像項目を読み込む（失敗時のみ従来の全走査へフォールバック）
        startup_items, image_paths = self._accessor.load_startup_image_items(
            startup_model_id
        )
        if startup_items:
            self._repository.set_image_paths(image_paths)
            self._image_items = startup_items
            self._logger.info(
                "起動時メタデータ読み込み: %s 件（DB由来）",
                len(startup_items),
            )
        else:
            self._logger.warning(
                "起動時メタデータをDBから取得できなかったため、ファイル全走査にフォールバックします。"
            )
            self._image_items = self._repository.load_all_image_item()

        # 画像IDと画像項目の対応を作成
        self._id_to_image_items = dict(map(lambda i: (i.id, i), self._image_items))

    def _get_index_and_items(
        self, model_id: ModelId, aesthetic_model_name: str
    ) -> tuple[Any, list[ImageItem]]:
        """指定モデル用のインデックスとメタ情報付き画像一覧を取得する"""

        return self._accessor.load_index_with_metadata(model_id, aesthetic_model_name)


    def _get_index_dimension(self, index, mean_vector: np.ndarray | None = None) -> int:
        """読み込んだFAISS indexまたは平均ベクトルから検索次元を取得する。"""

        index_dim = getattr(index, "d", None)
        if index_dim is not None:
            return int(index_dim)
        if mean_vector is not None:
            return int(mean_vector.size)
        raise ValueError("Unable to determine search embedding dimension from index or mean vector.")

    def _ensure_query_dimension(self, query_features: np.ndarray, index, mean_vector: np.ndarray | None = None) -> None:
        """クエリ次元とインデックス次元が一致することを検証する。"""

        expected_dim = self._get_index_dimension(index, mean_vector)
        actual_dim = int(query_features.shape[-1])
        if actual_dim != expected_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: query has {actual_dim} dimensions, "
                f"but the FAISS index expects {expected_dim}."
            )

    def _normalize_result_size(self, result_size: int | None) -> int:
        """検索結果件数の上限を安全な正整数に正規化する。"""

        try:
            if result_size is None:
                return 2048

            return max(1, int(result_size))
        except (TypeError, ValueError):
            return 2048

    def _sort_result_items(
        self, scores: list[ResultImageItem]
    ) -> list[ResultImageItem]:
        """検索結果をスコア降順、同点時は名前昇順で整列する。"""

        return sorted(
            scores,
            key=lambda result: (
                -float(result.score.score),
                result.item.display_name.name,
            ),
        )

    def _finalize_result(
        self,
        scores: list[ResultImageItem],
        search_query: str,
        result_size: int | None,
    ) -> ResultImageItemList:
        """JSON変換前の検索結果を整列し、指定件数に切り詰める。"""

        normalized_result_size = self._normalize_result_size(result_size)
        return ResultImageItemList(
            self._sort_result_items(scores)[:normalized_result_size], search_query
        )


    def warmup_search_cache(
        self,
        model_id: ModelId,
        search_text: str = "An image a cat.",
        aesthetic_model_name: str = "original",
    ) -> ResultImageItemList:
        """起動時に代表的なテキスト検索を実行し、検索に必要なデータをキャッシュする。

        実際のユーザー検索と同じ経路で埋め込みバックエンド、FAISSインデックス、
        画像メタデータ、平均ベクトルを読み込むことで、初回検索時の待ち時間を
        アプリケーション起動時に前倒しする。
        """

        self._logger.info(
            "起動時検索キャッシュを作成します: model_id=%s search_text=%s aesthetic_model_name=%s",
            model_id,
            search_text,
            aesthetic_model_name,
        )
        return self.search_text(
            model_id=model_id,
            text=UploadText(search_text),
            aesthetic_quality_beta=0.0,
            aesthetic_quality_range_min=0.0,
            aesthetic_quality_range_max=10.0,
            aesthetic_model_name=aesthetic_model_name,
            result_size=1,
        )

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

    def get_rating_list(
        self, model_id: ModelId, image_ids: list[ImageId] | None = None
    ) -> dict[ImageId, str]:
        """指定された画像IDに限定してrating一覧を取得する"""

        _, items = self._get_index_and_items(model_id, "original")
        target_ids = set(image_ids) if image_ids is not None else None
        return {
            item.id: item.rating
            for item in items
            if target_ids is None or item.id in target_ids
        }

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
            str: `np.array2string`でフォーマットした文字列表現。返却時に精度が抑制される。正規化してから文字列化することで、
                クエリの特徴量がどのような値の範囲にあるかをわかりやすくする。
        """
        f = search_query_obj.copy().astype(np.float32)
        faiss.normalize_L2(f)
        return np.array2string(f, separator=", ", suppress_small=True, threshold=np.inf, max_line_width=np.inf)

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
        self, item_list: list[ImageItem], index, query_features, result_size=8192,
        mean_centering=True, mean_vector=None
    ) -> list[ResultImageItem]:
        self._logger.info("query_features shape: %s", query_features.shape)
        self._ensure_query_dimension(query_features, index, mean_vector)

        # コピーして float32 を保証(faiss.normalize_L2 は in-place + float32 必須)
        q_features = query_features.copy().astype(np.float32)

        # ① 先に L2 正規化(mean_vector と同じスケールに揃える)
        faiss.normalize_L2(q_features)

        # ② Mean centering(後段で再正規化しない!)
        if mean_centering:
            if mean_vector is not None:
                if mean_vector.size != query_features.shape[-1]:
                    raise ValueError(
                        f"Mean vector dimension mismatch: mean vector has {mean_vector.size} "
                        f"dimensions, but query has {query_features.shape[-1]}."
                    )
                self._logger.info("Applying mean centering to query features.")
                q_features = q_features - mean_vector.reshape(1, -1).astype(np.float32)
            else:
                self._logger.warning("mean_vector が未指定または次元不一致のため中心化をスキップ")

        # ← ここに faiss.normalize_L2 を呼ばない

        item_list_length: int = len(item_list)
        if result_size > item_list_length:
            result_size = item_list_length

        index.nprobe = 64

        self._logger.info("index size: %s", index.ntotal)
        self._logger.info("query_features: %s", q_features)
        distances, indices = index.search(q_features, k=result_size)

        sorted_items: list[ImageItem] = sorted(item_list, key=lambda x: x.display_name.name)

        item_distances: dict[int, float] = {}
        for matched_indices, matched_distances in zip(indices, distances):
            for item_id, item_distance in zip(matched_indices, matched_distances):
                if item_id == -1:                       # FAISS は埋まらない枠を -1 で返す
                    continue
                item_distances[item_id] = item_distances.get(item_id, 0.0) + float(item_distance)

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
        result_size: int | None = None,
    ) -> ResultImageItemList:
        """文字列から検索する"""

        # テキストの埋め込みを計算
        backend = self._accessor.load_embedding_backend(model_id)
        features = backend.encode_text(text.text)
        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        mean_vector = self._accessor.get_mean_meta_vector(model_id)
        self._ensure_query_dimension(features, index, mean_vector)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            result_size=self._normalize_result_size(result_size),
            query_features=features,
            mean_centering=False,  # CLIPのテキスト特徴は中心化しない
            mean_vector=self._accessor.get_mean_meta_vector(model_id),
        )

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )

        return self._finalize_result(scores, self.format_search_query(features.copy()), result_size)


    def search_image(
        self,
        model_id: ModelId,
        id_list: list[ImageId],
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
        result_size: int | None = None,
    ) -> ResultImageItemList:
        """画像から検索する"""

        backend = self._accessor.load_embedding_backend(model_id)

        # 選択した画像のmetaを結合する
        temp = []
        for select_image_id in id_list:
            load_image = self._repository.load_image(select_image_id).to_ptl_image()
            temp.append(backend.encode_image(load_image))

        # クエリベクトルの平均を計算
        batch = np.vstack(temp)
        faiss.normalize_L2(batch)
        features = batch.mean(axis=0).reshape(1, -1)

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            result_size=self._normalize_result_size(result_size),
            query_features=features,
            mean_centering=True,  # CLIPの画像特徴は中心化する
            mean_vector=self._accessor.get_mean_meta_vector(model_id),
        )

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return self._finalize_result(scores, self.format_search_query(features), result_size)

    def search_name(
        self,
        model_id: ModelId,
        text: UploadText,
        is_regexp: bool,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
        result_size: int | None = None,
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
        return self._finalize_result(scores, "", result_size)

    def search_upload_image(
        self, model_id: ModelId, image: UploadImage, result_size: int | None = None
    ) -> ResultImageItemList:
        """アップロードされた画像から検索する"""

        backend = self._accessor.load_embedding_backend(model_id)

        # アップロードされた画像を前処理
        load_image = Image(binary=image.binary, content_type=image.content_type).to_ptl_image()
        features = backend.encode_image(load_image)

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, "original")

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            result_size=self._normalize_result_size(result_size),
            query_features=features,
            mean_centering=True,
            mean_vector=self._accessor.get_mean_meta_vector(model_id),
        )

        return self._finalize_result(scores, self.format_search_query(features), result_size)

    def search_random(
        self,
        model_id: ModelId,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
        result_size: int | None = None,
    ) -> ResultImageItemList:
        """乱数から検索する"""

<<<<<<< HEAD
        # 単位超球面（Unit hypersphere）上から一様にベクトルをサンプリング
        features: np.ndarray = np.random.normal(0, 1, [1, 768]).astype(np.float32)

=======
>>>>>>> a58619967598e25cec1f80703a15249ed73521b8
        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        mean_vector = self._accessor.get_mean_meta_vector(model_id)
        dimension = self._get_index_dimension(index, mean_vector)

        # 単位超球面（Unit hypersphere）上から一様にベクトルをサンプリング
        # similarity_evalの副作用で正規化されるため、ここでは正規化せずに生の乱数を渡す。
        features: np.ndarray = np.random.normal(0, 1, [1, dimension]).astype(np.float32)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            result_size=self._normalize_result_size(result_size),
            query_features=features,
            mean_centering=True,
            mean_vector=mean_vector,
        )
        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return self._finalize_result(scores, self.format_search_query(features), result_size)

    def search_query(
        self,
        model_id: ModelId,
        search_query: str,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
        result_size: int | None = None,
    ) -> ResultImageItemList:
        """クエリから検索する"""

        features = self.parse_search_query(search_query)
        if features is None:
            raise ValueError("Invalid search query vector format.")

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        mean_vector = self._accessor.get_mean_meta_vector(model_id)
        self._ensure_query_dimension(features, index, mean_vector)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            result_size=self._normalize_result_size(result_size),
            query_features=features,
            mean_centering=True,
            mean_vector=mean_vector,
        )

        scores: list[ResultImageItem] = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return self._finalize_result(scores, self.format_search_query(features), result_size)

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
        result_size: int | None = None,
    ) -> ResultImageItemList:
        """クエリにstrengthの強さ分テキストの特徴を足してから検索する"""

        backend = self._accessor.load_embedding_backend(model_id)

        query_features = self.parse_search_query(search_query)
        if query_features is None:
            raise ValueError("Invalid search query vector format.")

        text_features = backend.encode_text(text.text)

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        mean_vector = self._accessor.get_mean_meta_vector(model_id)
        self._ensure_query_dimension(query_features, index, mean_vector)
        self._ensure_query_dimension(text_features, index, mean_vector)

        faiss.normalize_L2(text_features)
        features = query_features + text_features * strength

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            result_size=self._normalize_result_size(result_size),
            query_features=features,
            mean_centering=True,
            mean_vector=mean_vector,
        )

        scores: list[ResultImageItem] = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name
        )
        return self._finalize_result(scores, self.format_search_query(features), result_size)


    def search_tags(
        self,
        model_id: ModelId,
        text: UploadText,
        is_regexp: bool,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
        result_size: int | None = None,
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
        return self._finalize_result(scores, "", result_size)

    def search_style_cluster(
        self,
        model_id: ModelId,
        text: UploadText,
        is_regexp: bool,
        aesthetic_quality_beta: float,
        aesthetic_quality_range_min: float,
        aesthetic_quality_range_max: float,
        aesthetic_model_name: str,
        result_size: int | None = None,
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
        return self._finalize_result(scores, "", result_size)

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
