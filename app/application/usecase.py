from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import BinaryIO, Literal

import faiss
import numpy as np
from PIL import Image as PILImage

from app.application.accessor import Accessor
from app.application.search import (
    SearchOptions,
    apply_aesthetics,
    check_dimension,
    nearest_neighbors,
    rank_results,
)
from app.application.search_index import SearchIndex
from app.application.text_matching import SearchMatcher
from app.domain import limits
from app.domain.domain_object import Image, ImageId, ImageItem, ModelId, ModelItem, SearchResults
from app.domain.errors import ImageUnavailableError, SearchInputError
from app.domain.rating import normalize_rating_filter
from app.domain.repository import Repository
from app.domain.search_query import InvalidSearchQuery, SearchQuery

logger = logging.getLogger(__name__)

MetadataField = Literal["name", "tags", "style_cluster"]

WARMUP_TEXT = "An image a cat."


@dataclass(frozen=True, slots=True)
class ImageArchive:
    """ZIPのストリームと、要求に対して実際に含められた画像の件数。"""

    stream: BinaryIO
    requested_count: int
    image_count: int
    skipped_count: int


def open_search_image(image: Image) -> PILImage.Image:
    """検索に使う画像を展開する。容量・画素数の上限は展開の前に確認される。"""
    try:
        return image.to_pil_image()
    except (OSError, ValueError, SyntaxError) as error:
        raise SearchInputError("画像を読み込めません。別の画像を選んでください。") from error


class Usecase:
    """このアプリケーションの動作を実装するクラス"""

    def __init__(self, repository: Repository, accessor: Accessor) -> None:
        self._repository = repository
        self._accessor = accessor

    # ===================================================================
    # 一覧・画像

    def get_all_model(self) -> list[ModelItem]:
        """すべての検索モデルを取得する"""
        return self._repository.load_all_model_item()

    def get_image_items_by_page(
        self, model_id: ModelId, page: int, size: int, ratings: Sequence[str] | None = None,
    ) -> list[ImageItem]:
        """ページングして画像項目を取得する"""
        return self._repository.load_image_items_by_page(
            model_id, page, size, normalize_rating_filter(ratings)
        )

    def get_catalog_count(self, model_id: ModelId, ratings: Sequence[str] | None = None) -> int:
        """区分に一致する画像項目の件数を取得する"""
        return self._repository.count_image_items(model_id, normalize_rating_filter(ratings))

    def get_image_metadata(self, model_id: ModelId, image_id: ImageId) -> ImageItem | None:
        """画像のタグや評価などのメタデータを取得する"""
        return self._repository.find_image_item(model_id, image_id)

    def get_image(self, image_id: ImageId) -> Image:
        """画像IDから画像項目の画像を取得する"""
        return self._repository.load_image(image_id)

    def get_small_image(self, image_id: ImageId) -> Image:
        """画像IDから縮小画像を取得する"""
        return self._repository.load_small_image(image_id)

    # ===================================================================
    # 検索

    def warmup_search(self, model_id: ModelId) -> None:
        """起動時に代表的なテキスト検索を実行し、検索に必要なデータを読み込んでおく。

        実際のユーザー検索と同じ経路で埋め込みバックエンド・FAISSインデックス・
        メタデータを読み込むことで、初回検索時の待ち時間を起動時に前倒しする。
        """
        logger.info("起動時検索キャッシュを作成します: model_id=%s", model_id)
        self.search_text(model_id, WARMUP_TEXT, SearchOptions(result_size=1))

    def search_text(self, model_id: ModelId, text: str, options: SearchOptions) -> SearchResults:
        """文字列から検索する"""
        features = self._accessor.load_embedding_backend(model_id).encode_text(text)
        search_index = self._accessor.load_search_index(model_id)
        # CLIPのテキスト特徴は中心化しない
        return self._search_vector(model_id, search_index, features, False, options)

    def search_image(
        self, model_id: ModelId, image_ids: Sequence[ImageId], options: SearchOptions,
    ) -> SearchResults:
        """登録済みの画像から検索する。複数指定した場合は特徴量の平均を使う。"""
        vectors = []
        for image_id in image_ids:
            try:
                image = self._repository.load_image(image_id)
            except ImageUnavailableError as error:
                raise SearchInputError(f"選択画像を読み込めません: {image_id}") from error
            vectors.append(self._encode_image(model_id, image))

        batch = np.vstack(vectors)
        faiss.normalize_L2(batch)
        features = batch.mean(axis=0).reshape(1, -1)
        search_index = self._accessor.load_search_index(model_id)
        return self._search_vector(model_id, search_index, features, True, options)

    def search_upload_image(self, model_id: ModelId, image: Image, options: SearchOptions) -> SearchResults:
        """アップロードされた画像から検索する"""
        features = self._encode_image(model_id, image)
        search_index = self._accessor.load_search_index(model_id)
        return self._search_vector(model_id, search_index, features, True, options)

    def search_random(self, model_id: ModelId, options: SearchOptions) -> SearchResults:
        """乱数から検索する"""
        search_index = self._accessor.load_search_index(model_id)
        # 正規分布の乱数は、検索時の正規化で単位超球面上の一様な方向になる。
        features = np.random.normal(0, 1, [1, search_index.dimension]).astype(np.float32)
        return self._search_vector(model_id, search_index, features, True, options)

    def search_query(self, model_id: ModelId, query: SearchQuery, options: SearchOptions) -> SearchResults:
        """保存クエリから検索する"""
        search_index, features = self._load_saved_query(model_id, query)
        return self._search_vector(model_id, search_index, features, query.mean_centering, options)

    def add_text_features(
        self, model_id: ModelId, text: str, query: SearchQuery, strength: float, options: SearchOptions,
    ) -> SearchResults:
        """保存クエリにstrengthの強さ分テキストの特徴を足してから検索する"""
        search_index, features = self._load_saved_query(model_id, query)
        # A no-op refinement preserves the original float32 input exactly.
        if strength != 0:
            text_features = self._accessor.load_embedding_backend(model_id).encode_text(text)
            check_dimension(search_index, text_features)
            # Slider strength is relative to direction, not the source's norm.
            faiss.normalize_L2(text_features)
            faiss.normalize_L2(features)
            features = features + text_features * strength
        return self._search_vector(model_id, search_index, features, query.mean_centering, options)

    def search_metadata(
        self, model_id: ModelId, field: MetadataField, text: str, is_regexp: bool, options: SearchOptions,
    ) -> SearchResults:
        """名前・タグ・style_clusterのいずれかを文字列検索する"""
        matcher = SearchMatcher(text, is_regexp)
        search_index = self._accessor.load_search_index(model_id)
        values = {
            "name": search_index.names,
            "tags": search_index.tags,
            "style_cluster": search_index.style_clusters,
        }[field]
        positions = matcher.matching_positions(values)
        positions, scores = apply_aesthetics(search_index, positions, np.ones(len(positions)), options)
        return self._results(search_index, positions, scores, options, "")

    def _encode_image(self, model_id: ModelId, image: Image) -> np.ndarray:
        # 画像の検証は、埋め込みモデルを読み込む前に済ませる。
        with open_search_image(image) as decoded:
            return self._accessor.load_embedding_backend(model_id).encode_image(decoded)

    def _load_saved_query(self, model_id: ModelId, query: SearchQuery) -> tuple[SearchIndex, np.ndarray]:
        if query.model_id != model_id:
            saved = query.model_id
            raise InvalidSearchQuery(
                "search_query was saved for a different model; select "
                f"{saved.model_name} ({saved.pretrained or 'no pretrained tag'})"
            )
        features = np.array(query.vector, dtype=np.float32).reshape(1, -1)
        search_index = self._accessor.load_search_index(model_id)
        check_dimension(search_index, features, error_type=InvalidSearchQuery)
        return search_index, features

    def _search_vector(
        self,
        model_id: ModelId,
        search_index: SearchIndex,
        features: np.ndarray,
        mean_centering: bool,
        options: SearchOptions,
    ) -> SearchResults:
        positions, scores = nearest_neighbors(
            search_index, features, mean_centering=mean_centering, options=options
        )
        positions, scores = apply_aesthetics(search_index, positions, scores, options)
        # 保存時に正規化や丸めを重ねず、初回検索と再検索で同じfloat32入力を使う。
        vector = tuple(features.astype(np.float32).reshape(-1).tolist())
        return self._results(
            search_index, positions, scores, options, SearchQuery(vector, mean_centering, model_id).to_text()
        )

    def _results(
        self,
        search_index: SearchIndex,
        positions: np.ndarray,
        scores: np.ndarray,
        options: SearchOptions,
        search_query: str,
    ) -> SearchResults:
        results = rank_results(search_index, positions, scores, options.result_size)
        # 結果の画像はすぐに取得されるため、IDと場所の対応を先に渡しておく。
        self._repository.remember_image_items(result.item for result in results)
        return SearchResults(results, search_query)

    # ===================================================================
    # 保存

    def get_download_ids(
        self, model_id: ModelId, limit: int, ratings: Sequence[str] | None = None,
    ) -> list[ImageId]:
        """一覧と同じ順序で、先頭から最大limit件の画像IDを取得する"""
        if type(limit) is not int or not 1 <= limit <= limits.MAX_DOWNLOAD_IMAGES:
            raise SearchInputError(f"Download limit must be from 1 to {limits.MAX_DOWNLOAD_IMAGES}")
        items = self._repository.load_image_items_by_page(model_id, 0, limit, normalize_rating_filter(ratings))
        return [item.id for item in items]

    def get_images_zip(self, image_ids: Sequence[ImageId]) -> ImageArchive:
        """指定された画像をZIPにまとめる。読めない画像は記録して除外する。

        画像は書き込みの直前に1件ずつ読み込む。ストリームは呼び出し側がcloseする。
        """
        logger.info("start:get_images_zip")
        image_count = 0

        def images():
            nonlocal image_count
            for image_id in image_ids:
                try:
                    image = self._repository.load_image(image_id)
                except ImageUnavailableError as error:
                    logger.warning("画像を読み込めないためスキップ: %s", error)
                    continue
                image_count += 1
                yield image

        stream = self._repository.create_zip_from_images(images())
        if not image_count:
            stream.close()
            raise SearchInputError("保存できる画像がありません。画像の場所と検索条件を確認してください。")
        return ImageArchive(stream, len(image_ids), image_count, len(image_ids) - image_count)
