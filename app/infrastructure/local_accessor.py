from functools import cache
import logging
import sqlite3
from typing import Any, Dict, List, Tuple

import pathlib

import numpy as np
import pandas as pd
import tqdm
import faiss
import open_clip

from app.domain.domain_object import (
    ImageId,
    ImageItem,
    ImageName,
    ImageTags,
    ModelId,
    Model,
    Tokenizer,
)
from app.application.accessor import Accessor


class LocalAccessor(Accessor):
    """ローカル上のファイルを対象としたAccessor"""

    def __init__(self, meta_dir_path) -> None:
        self._logger = logging.getLogger(__name__)
        self._meta_dir_path = meta_dir_path
        self._id_to_path: Dict[ImageId, pathlib.Path] = {}
        self._mean_vector_cache: np.ndarray | None = None

    def load_image_feature(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        """meta_dir配下のmodel-pretrained/image_id.npyから画像特徴量を読み込み、毎回np.loadで返す（キャッシュ無し）。"""
        return np.load(
            f"{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/{image_id}.npy"
        )

    @cache
    def load_model(self, model_id: ModelId) -> Model:
        """モデル名とpretrained設定でopen_clipモデル+変換器を構築しModelに包んで返す（結果は@cacheで共有）。"""

        model_name, pretrained = model_id.model_name, model_id.pretrained
        return Model(
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        )

    @cache
    def load_tokenizer(self, model_id: ModelId) -> Tokenizer:
        """model_id.model_nameに対応するopen_clipトークナイザを生成し、@cacheで同一インスタンスを再利用する。"""
        return Tokenizer(open_clip.get_tokenizer(model_id.model_name))

    @cache
    def load_index_with_metadata(
        self, model_id: ModelId, aesthetic_model_name: str
    ) -> Tuple[Any, List[ImageItem]]:
        """meta_dir/model-pretrainedのFAISS indexとSQLiteメタを読み込み、画像パス順のImageItemリストとindexをタプルで返す（@cacheで結果共有）。"""
        index = faiss.read_index(
            f'{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/{"metafiles.index"}'
        )

        con: sqlite3.Connection = sqlite3.connect(
            f"{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/sqlite_image_meta.db",
            isolation_level="DEFERRED",
        )
        result: pd.DataFrame = pd.read_sql_query(
            """
            SELECT image_id, image_path, image_tags, aesthetic_quality, pony_aesthetic_quality, style_cluster, rating
            FROM image_meta
            """,
            con,
        )
        con.close()

        aesthetic_name = (
            "pony_aesthetic_quality" if aesthetic_model_name != "original" else "aesthetic_quality"
        )
        sorted_result: pd.DataFrame = result.sort_values("image_path").reset_index(
            drop=True
        )

        image_items: List[ImageItem] = []
        for row in tqdm.tqdm(sorted_result.itertuples(index=False)):
            aesthetic_score = getattr(row, aesthetic_name, None)
            image_items.append(
                ImageItem(
                    id=ImageId(str(row.image_id)),
                    display_name=ImageName(row.image_path),
                    tags=ImageTags(row.image_tags),
                    aesthetic_quality=float(aesthetic_score)
                    if aesthetic_score is not None
                    else None,
                    rating=row.rating or "",
                    style_cluster=row.style_cluster or "",
                )
            )

        return index, image_items

    @cache
    def load_startup_image_items(
        self,
        model_id: ModelId,
    ) -> tuple[list[ImageItem], dict[ImageId, pathlib.Path]]:
        """起動時に必要な画像一覧とImageId->相対パスをSQLiteから構築して返す。"""

        startup_items_by_id: dict[ImageId, ImageItem] = {}
        id_to_path: dict[ImageId, pathlib.Path] = {}
        db_path = (
            self._meta_dir_path
            / f"{model_id.model_name}-{model_id.pretrained}"
            / "sqlite_image_meta.db"
        )

        if not db_path.exists():
            self._logger.warning(
                "起動用のsqlite_image_meta.dbが見つかりません: %s",
                db_path,
            )
            return [], {}

        con: sqlite3.Connection = sqlite3.connect(
            db_path,
            isolation_level="DEFERRED",
        )
        try:
            result: pd.DataFrame = pd.read_sql_query(
                """
                SELECT image_id, image_path, image_tags
                FROM image_meta
                """,
                con,
            )
        finally:
            con.close()

        for row in result.itertuples(index=False):
            image_id = ImageId(str(row.image_id))
            if image_id in startup_items_by_id:
                continue

            image_path = pathlib.Path(str(row.image_path))
            tags = row.image_tags if row.image_tags is not None else ""
            startup_items_by_id[image_id] = ImageItem(
                id=image_id,
                display_name=ImageName(str(image_path)),
                tags=ImageTags(tags),
            )
            id_to_path[image_id] = image_path

        startup_items = sorted(
            startup_items_by_id.values(),
            key=lambda item: item.display_name.name,
        )
        return startup_items, id_to_path

    def get_mean_meta_vector(self, model_id: ModelId) -> np.ndarray | None:
        """ModelIdからimage meta全体の平均ベクトルを取得する。"""

        if hasattr(self, "mean_vector_cache") and self._mean_vector_cache is not None:
            return self._mean_vector_cache

        mean_vector = np.load(
                f"{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/metafiles.index.mean.npy"
            )

        # キャッシュに保存
        self._mean_vector_cache = mean_vector
        return mean_vector
