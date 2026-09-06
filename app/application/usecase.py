import logging
import traceback
from typing import Any
import numpy as np
import faiss
import tqdm
from PIL import Image as PILImage

from app.application.accessor import Accessor
from app.application.text_matching import SearchMatcher
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
from app.domain.errors import ResourceLimitError, SearchInputError
from app.domain.search_query import InvalidSearchQuery, SearchQuery, parse_search_query


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
        self._aesthetic_score_arrays = {}
        self._startup_model_id = startup_model_id
        self._paged_catalog = False

        # A large catalog must not be materialized merely to open the gallery.
        first_page = self._accessor.load_startup_image_page(startup_model_id, 0, 60)
        if first_page is not None:
            startup_items, image_paths = first_page
            self._paged_catalog = True
            self._repository.set_image_paths(image_paths)
            self._id_to_image_items.update((item.id, item) for item in startup_items)
            self._logger.info("起動時メタデータ読み込み: 先頭ページ %s 件（以降は必要時取得）", len(startup_items))
            return

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
        """インデックスと、その行番号に対応する順序の画像一覧を取得する。"""

        return self._accessor.load_index_with_metadata(model_id, aesthetic_model_name)


    def _get_index_dimension(self, index, mean_vector: np.ndarray | None = None) -> int:
        """読み込んだFAISS indexまたは平均ベクトルから検索次元を取得する。"""

        index_dim = getattr(index, "d", None)
        if index_dim is not None:
            return int(index_dim)
        if mean_vector is not None:
            return int(mean_vector.size)
        raise ValueError("Unable to determine search embedding dimension from index or mean vector.")

    def _ensure_query_dimension(self, query_features: np.ndarray, index, mean_vector: np.ndarray | None = None, *, error_type=ValueError) -> None:
        """クエリ次元とインデックス次元が一致することを検証する。"""

        expected_dim = self._get_index_dimension(index, mean_vector)
        actual_dim = int(query_features.shape[-1])
        if actual_dim != expected_dim:
            raise error_type(
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
        selected = self._sort_result_items(scores)[:normalized_result_size]
        items = [result.item for result in selected]
        self._register_image_items(items)
        return ResultImageItemList(selected, search_query)

    def _register_image_items(self, items: list[ImageItem]) -> None:
        self._repository.register_image_items(items)
        self._id_to_image_items.update((item.id, item) for item in items)


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
        """全件取得を明示した内部呼出用。HTTPの一覧はページ取得を使う。"""

        if self._paged_catalog:
            items, _paths = self._accessor.load_startup_image_items(self._startup_model_id)
            self._register_image_items(items)
            return items
        return list(self._image_items)

    def get_image_items_by_page(self, page: int, size: int, model_id: ModelId | None = None, ratings: list[str] | None = None) -> list[ImageItem]:
        """ページングして画像項目を取得する"""

        if size <= 0:
            return []

        normalized_page = max(page, 0)
        if self._paged_catalog or model_id is not None or ratings is not None:
            args = (model_id or self._startup_model_id, normalized_page, size)
            result = (self._accessor.load_startup_image_page(*args, ratings)
                      if ratings is not None else self._accessor.load_startup_image_page(*args))
            if result is None:
                raise ValueError("画像一覧のDBを読み込めません。データの場所を確認してください。")
            items, _paths = result
            self._register_image_items(items)
            return items
        start_index = normalized_page * size
        end_index = start_index + size

        return self._image_items[start_index:end_index]

    def get_catalog_count(self, model_id: ModelId | None = None, ratings: list[str] | None = None) -> int | None:
        model_id = model_id or self._startup_model_id
        return (self._accessor.load_catalog_count(model_id, ratings)
                if ratings is not None else self._accessor.load_catalog_count(model_id))

    def get_image_item(self, id: ImageId) -> ImageItem:
        """画像IDから画像項目を取得する"""

        self._resolve_image_id(id)
        return self._id_to_image_items[id]

    def _resolve_image_id(self, image_id: ImageId) -> None:
        """保存済み画像URLも、一覧の閲覧順によらず1件のDB参照で解決する。"""
        if not getattr(self, "_paged_catalog", False) or image_id in self._id_to_image_items:
            return
        models = [self._startup_model_id]
        models.extend(item.id for item in self._repository.load_all_model_item() if item.id not in models)
        for model_id in models:
            items = self._accessor.load_image_metadata(model_id, [image_id])
            if items:
                self._register_image_items(items)
                return
        raise ValueError(f"画像が見つかりません: {image_id.id}")

    def get_image(self, id: ImageId) -> Image:
        """画像IDから画像項目の画像を取得する"""

        self._resolve_image_id(id)
        return self._repository.load_image(id)

    def get_small_image(self, id: ImageId) -> Image:
        """画像IDから縮小画像を取得する"""

        self._resolve_image_id(id)
        return self._repository.load_small_image(id)

    def get_image_metadata(self, model_id: ModelId, image_id: ImageId) -> dict[str, str | float]:
        """画像のタグや評価などのメタデータを取得する"""

        items = self._accessor.load_image_metadata(model_id, [image_id])
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

        items = self._accessor.load_image_metadata(model_id, image_ids)
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

    def format_search_query(self, search_query_obj, model_id: ModelId | None = None, *, mean_centering: bool = True) -> str:
        """検索クエリのnumpy配列を文字列表現に整形する。

        Args:
            search_query_obj (np.ndarray): 検索クエリとして扱う特徴量配列。1行ベクトルを想定。

        Returns:
            str: float32精度の特徴量・モデル・中心化条件を保持するJSON文字列。
                保存時に正規化や丸めを重ねず、初回検索と再検索で同じ入力を使う。
                model_id省略時は旧形式のベクトル文字列を返す。
        """
        vector = search_query_obj.astype(np.float32).reshape(-1).tolist()
        return SearchQuery(tuple(vector), mean_centering, model_id).to_text()

    def parse_search_query(self, search_query_text: str):
        """文字列表現の検索クエリをnumpy配列に変換する。

        Args:
            search_query_text (str): 新形式の保存クエリ、または旧形式のベクトル文字列。

        Returns:
            np.ndarray: float32の1行ベクトル。不正入力にはInvalidSearchQueryを送出する。
        """
        return np.array(parse_search_query(search_query_text).vector, dtype=np.float32).reshape(1, -1)

    # ===================================================================

    @staticmethod
    def _aesthetic_search_range(beta, low, high):
        return (low, high) if beta != 0 or low > 0 or high < 10 else None

    def _aesthetic_selector(self, item_list, quality_range):
        """Cache scores once; search only eligible IDs using a compact bitmap."""
        if not hasattr(self, "_aesthetic_score_arrays"):
            self._aesthetic_score_arrays = {}
        cache_key = id(item_list)
        cached = self._aesthetic_score_arrays.get(cache_key)
        if cached is None or cached[0] is not item_list:
            values = np.fromiter(
                (item.aesthetic_quality if item.aesthetic_quality is not None else np.nan for item in item_list),
                dtype=np.float64, count=len(item_list),
            )
            self._aesthetic_score_arrays[cache_key] = (item_list, values)
        else:
            values = cached[1]
        eligible = np.isfinite(values) & (values >= quality_range[0]) & (values <= quality_range[1])
        count = int(np.count_nonzero(eligible))
        return faiss.IDSelectorBitmap(np.packbits(eligible, bitorder="little")), count

    def similarity_eval(
        self, item_list: list[ImageItem], index, query_features, result_size=8192,
        mean_centering=True, mean_vector=None, aesthetic_range=None,
    ) -> list[ResultImageItem]:
        self._logger.info("query_features shape: %s", query_features.shape)
        self._ensure_query_dimension(query_features, index, mean_vector)
        if index.ntotal != len(item_list):
            raise ValueError(
                f"Search index contains {index.ntotal} vectors, but metadata contains "
                f"{len(item_list)} images. Rebuild the index and metadata together."
            )

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

        result_size = min(
            self._normalize_result_size(result_size), len(item_list), index.ntotal
        )
        if result_size == 0:
            return []

        parameters = None
        if aesthetic_range is not None:
            selector, eligible_count = self._aesthetic_selector(item_list, aesthetic_range)
            result_size = min(result_size, eligible_count)
            if result_size == 0:
                return []
            parameters = (
                faiss.SearchParametersIVF(nprobe=64, sel=selector)
                if hasattr(index, "nprobe") else faiss.SearchParameters(sel=selector)
            )
        elif hasattr(index, "nprobe"):
            index.nprobe = 64

        self._logger.info("index size: %s", index.ntotal)
        self._logger.info("query_features: %s", q_features)
        if parameters is None:
            distances, indices = index.search(q_features, k=result_size)
        else:
            distances, indices = index.search(q_features, k=result_size, params=parameters)

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
                    # Accessor が FAISS の行番号順に返したメタデータを直接参照する。
                    ResultImageItem(item_list[item_id], Score(total_distance))
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
            list[ResultImageItem]: 評価が有効な場合は範囲内の項目だけを再スコアリングした結果。
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
            if not aesthetic_quality_range_min <= aesthetic_quality_score <= aesthetic_quality_range_max:
                continue
            new_score = (
                i.score.score * (1 - aesthetic_quality_beta**2)
                + aesthetic_quality_score * aesthetic_quality_beta
            )
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
            aesthetic_range=self._aesthetic_search_range(aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max),
        )

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )

        return self._finalize_result(scores, self.format_search_query(features, model_id, mean_centering=False), result_size)


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

        # 選択した画像のmetaを結合する
        temp = []
        for select_image_id in id_list:
            try:
                image = self.get_image(select_image_id)
            except (ValueError, FileNotFoundError) as error:
                raise SearchInputError(f"選択画像を読み込めません: {select_image_id.id}") from error
            with self._open_search_image(image) as load_image:
                backend = self._accessor.load_embedding_backend(model_id)
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
            aesthetic_range=self._aesthetic_search_range(aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max),
        )

        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return self._finalize_result(scores, self.format_search_query(features, model_id), result_size)

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
        matches = SearchMatcher(text.text, is_regexp)
        # 類似度を計算する
        _, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        for image_item in tqdm.tqdm(item_list):
            name: str = image_item.display_name.name
            # print(args.get("trueRegexp"))
            if matches(name):
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

    @staticmethod
    def _open_search_image(image: Image):
        """画像検索でも展開前の画素上限と読み込みエラーを確認する。"""
        decoded = None
        loaded = False
        try:
            decoded = image.to_ptl_image()
            if decoded.width * decoded.height > 64_000_000:
                raise ResourceLimitError("検索画像は6400万画素以内で指定してください。")
            decoded.load()
            loaded = True
            return decoded
        except PILImage.DecompressionBombError:
            raise ResourceLimitError("検索画像は6400万画素以内で指定してください。") from None
        except (OSError, ValueError, SyntaxError) as error:
            raise SearchInputError("画像を読み込めません。別の画像を選んでください。") from error
        finally:
            # Failed decoding must not leave an open image/file behind.
            if decoded is not None and not loaded:
                decoded.close()

    def search_upload_image(
        self, model_id: ModelId, image: UploadImage, result_size: int | None = None
    ) -> ResultImageItemList:
        """アップロードされた画像から検索する"""

        # アップロードされた画像を前処理
        with self._open_search_image(Image(binary=image.binary, content_type=image.content_type)) as load_image:
            backend = self._accessor.load_embedding_backend(model_id)
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

        return self._finalize_result(scores, self.format_search_query(features, model_id), result_size)

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
            aesthetic_range=self._aesthetic_search_range(aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max),
        )
        scores = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return self._finalize_result(scores, self.format_search_query(features, model_id), result_size)

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

        query = parse_search_query(search_query, model_id)
        features = np.array(query.vector, dtype=np.float32).reshape(1, -1)

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        mean_vector = self._accessor.get_mean_meta_vector(model_id)
        self._ensure_query_dimension(features, index, mean_vector, error_type=InvalidSearchQuery)

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            result_size=self._normalize_result_size(result_size),
            query_features=features,
            mean_centering=query.mean_centering,
            mean_vector=mean_vector,
            aesthetic_range=self._aesthetic_search_range(aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max),
        )
        # Range filtering happens inside FAISS before the result limit is applied.

        scores: list[ResultImageItem] = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name,
        )
        return self._finalize_result(scores, self.format_search_query(features, model_id, mean_centering=query.mean_centering), result_size)

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

        query = parse_search_query(search_query, model_id)
        query_features = np.array(query.vector, dtype=np.float32).reshape(1, -1)

        # indexを読み込み
        index, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        mean_vector = self._accessor.get_mean_meta_vector(model_id)
        self._ensure_query_dimension(query_features, index, mean_vector, error_type=InvalidSearchQuery)
        if strength == 0:
            # Preserve the original float32 input exactly for a no-op refinement.
            features = query_features
        else:
            backend = self._accessor.load_embedding_backend(model_id)
            text_features = backend.encode_text(text.text)
            self._ensure_query_dimension(text_features, index, mean_vector)
            faiss.normalize_L2(text_features)
            # Slider strength is relative to direction, not the source's norm.
            faiss.normalize_L2(query_features)
            features = query_features + text_features * strength

        # 類似度を計算する
        scores: list[ResultImageItem] = self.similarity_eval(
            item_list=item_list,
            index=index,
            result_size=self._normalize_result_size(result_size),
            query_features=features,
            mean_centering=query.mean_centering,
            mean_vector=mean_vector,
            aesthetic_range=self._aesthetic_search_range(aesthetic_quality_beta, aesthetic_quality_range_min, aesthetic_quality_range_max),
        )

        scores: list[ResultImageItem] = self.apply_aesthetic_quality_filter(
            model_id,
            scores,
            aesthetic_quality_beta,
            aesthetic_quality_range_min,
            aesthetic_quality_range_max,
            aesthetic_model_name
        )
        return self._finalize_result(scores, self.format_search_query(features, model_id, mean_centering=query.mean_centering), result_size)


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
        matches = SearchMatcher(text.text, is_regexp)
        # 類似度を計算する
        _, item_list = self._get_index_and_items(model_id, aesthetic_model_name)
        for image_item in tqdm.tqdm(item_list):
            tags: str = image_item.tags.tags
            # print(args.get("trueRegexp"))
            if matches(tags):
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
        matches = SearchMatcher(text.text, is_regexp)
        _, item_list = self._get_index_and_items(model_id, aesthetic_model_name)

        for image_item in tqdm.tqdm(item_list):
            style_cluster = image_item.style_cluster

            if matches(style_cluster):
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

    def get_download_ids(self, model_id: ModelId, limit: int, ratings: list[str] | None = None) -> list[str]:
        """Resolve a bounded selection in the same model/path order as browsing."""
        items = self._accessor.load_download_image_items(model_id, limit, ratings)
        self._register_image_items(items)
        return [item.id.id for item in items]

    def get_images_zip(self, id_list):
        """指定された画像IDからZIPバッファを生成する。

        Args:
            id_list (Iterable[str]): 画像IDの反復可能オブジェクト。各IDは`ImageId`へ変換される。

        Returns:
            BinaryIO: 取得できた画像のZIPを保持する、一時ファイルへ退避可能なバッファ。呼び出し側がcloseする。無効IDはログに記録してスキップする。
        """
        self._logger.info("start:get_images_zip")
        counts = {"requested_count": 0, "image_count": 0, "skipped_count": 0}
        def images_with_names():
            for image_id_str in tqdm.tqdm(id_list):
                counts["requested_count"] += 1
                image_id = ImageId(image_id_str)
                try:
                    image = self.get_image(image_id)
                    image_name = self._repository.get_image_name(image_id)
                    if image is None:
                        self._logger.warning("画像IDの読み込み結果がNoneのためスキップ: %s", image_id)
                        counts["skipped_count"] += 1
                        continue
                    counts["image_count"] += 1
                    yield image, image_name
                except (ValueError, FileNotFoundError) as e:
                    counts["skipped_count"] += 1
                    self._logger.warning("画像IDが無効のためスキップ: %s", e)
            if not counts["image_count"]:
                raise SearchInputError("保存できる画像がありません。画像の場所と検索条件を確認してください。")

        zip_buffer = self._repository.create_zip_from_images(images_with_names())
        # BinaryIO implementations used by HTTP can expose useful preparation
        # statistics while preserving the existing stream-returning interface.
        for name, value in counts.items():
            try:
                setattr(zip_buffer, name, value)
            except AttributeError:
                pass
        return zip_buffer
# ===================================================================
