from collections import OrderedDict
from collections.abc import Iterable
import logging
import pathlib
import sqlite3
import threading
from typing import BinaryIO

from app.domain.domain_object import Image, ImageId, ImageItem, ModelId, ModelItem
from app.domain.errors import ImageUnavailableError
from app.domain.rating import RatingFilter
from app.domain.repository import Repository
from app.infrastructure.catalog_database import CatalogDatabase
from app.infrastructure.image_store import DEFAULT_THUMBNAIL_CACHE_BYTES, ImageStore
from app.infrastructure.model_registry import ModelRegistry
from app.infrastructure.zip_archive import create_zip

logger = logging.getLogger(__name__)

_MAX_REMEMBERED_PATHS = 100_000


class LocalRepository(Repository):
    """ローカル上のメタデータDBと画像ファイルを対象としたRepository"""

    def __init__(
        self,
        image_dir: pathlib.Path,
        registry: ModelRegistry,
        *,
        thumbnail_cache_bytes: int = DEFAULT_THUMBNAIL_CACHE_BYTES,
    ) -> None:
        self._registry = registry
        self._catalog = CatalogDatabase(registry)
        self._images = ImageStore(image_dir, thumbnail_cache_bytes=thumbnail_cache_bytes)
        # 画像ID -> 画像ルートからの相対パス。一覧に出した画像と、DBから解決した画像を
        # 利用の古い順に上限まで保持し、画像配信ごとのDB参照を省く。
        self._paths: OrderedDict[ImageId, str] = OrderedDict()
        # 画像IDを探す順序。直近に見つかったモデルを先頭にする。
        self._lookup_order = [item.id for item in registry.items()]
        self._unreadable_models: set[ModelId] = set()
        self._path_lock = threading.Lock()

    def load_all_model_item(self) -> list[ModelItem]:
        return self._registry.items()

    def count_image_items(self, model_id: ModelId, ratings: RatingFilter = None) -> int:
        return self._catalog.count(model_id, ratings)

    def load_image_items_by_page(
        self, model_id: ModelId, page: int, size: int, ratings: RatingFilter = None,
    ) -> list[ImageItem]:
        items = self._catalog.page(model_id, page, size, ratings)
        with self._path_lock:
            for item in items:
                self._remember_path(item.id, item.name)
        return items

    def find_image_item(self, model_id: ModelId, image_id: ImageId) -> ImageItem | None:
        return self._catalog.find(model_id, image_id)

    def load_image(self, image_id: ImageId) -> Image:
        return self._images.load_original(self._path(image_id))

    def load_small_image(self, image_id: ImageId) -> Image:
        return self._images.load_thumbnail(self._path(image_id))

    def remember_image_items(self, items: Iterable[ImageItem]) -> None:
        with self._path_lock:
            for item in items:
                if item.name:
                    self._remember_path(item.id, item.name)

    def create_zip_from_images(self, images: Iterable[Image]) -> BinaryIO:
        return create_zip(images)

    def _remember_path(self, image_id: ImageId, path: str) -> None:
        self._paths[image_id] = path
        self._paths.move_to_end(image_id)
        while len(self._paths) > _MAX_REMEMBERED_PATHS:
            self._paths.popitem(last=False)

    def _path(self, image_id: ImageId) -> str:
        """保存済みの画像URLも、一覧の閲覧順によらず主キーの参照だけで解決する。"""
        with self._path_lock:
            path = self._paths.get(image_id)
            if path is not None:
                self._paths.move_to_end(image_id)
                return path
            lookup_order = list(self._lookup_order)
        for model_id in lookup_order:
            try:
                path = self._catalog.find_path(model_id, image_id)
            except sqlite3.Error as error:
                self._demote_unreadable_model(model_id, error)
                continue
            if path is not None:
                with self._path_lock:
                    self._remember_path(image_id, path)
                    self._lookup_order.remove(model_id)
                    self._lookup_order.insert(0, model_id)
                    self._unreadable_models.discard(model_id)
                return path
        raise ImageUnavailableError(f"画像IDが見つかりません: {image_id}")

    def _demote_unreadable_model(self, model_id: ModelId, error: sqlite3.Error) -> None:
        """1つのモデルのDBを読めなくても、ほかのモデルの画像は返す。読めないモデルは最後に試す。"""
        with self._path_lock:
            self._lookup_order.remove(model_id)
            self._lookup_order.append(model_id)
            first_failure = model_id not in self._unreadable_models
            self._unreadable_models.add(model_id)
        if first_failure:
            logger.warning("画像IDの参照でメタデータDBを読めません: %s (%s)", model_id, error)
