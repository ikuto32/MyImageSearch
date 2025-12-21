from functools import cache
import sqlite3
from typing import Any, Dict, List

import pathlib
import sqlite3
from typing import Any, Dict

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

        self._meta_dir_path = meta_dir_path
        self._id_to_path: Dict[ImageId, pathlib.Path] = {}

    def load_meta(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        return np.load(
            f"{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/{image_id}.npy"
        )

    @cache
    def load_index_file(self, model_id: ModelId) -> Any:

        index = faiss.read_index(
            f'{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/{"metafiles.index"}'
        )
        return index

    @cache
    def load_model(self, model_id: ModelId) -> Model:

        model_name, pretrained = model_id.model_name, model_id.pretrained
        return Model(
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        )

    @cache
    def load_tokenizer(self, model_id: ModelId) -> Tokenizer:
        return Tokenizer(open_clip.get_tokenizer(model_id.model_name))

    @cache
    def load_index_item_list(self, model_id: ModelId) -> List[ImageItem]:
        print("start:load_index_item_list")
        con: sqlite3.Connection = sqlite3.connect(
            f"{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/sqlite_image_meta.db",
            isolation_level="DEFERRED",
        )
        result: pd.DataFrame = pd.read_sql_query(
            """
            SELECT image_id, image_path, image_tags FROM image_meta
            """,
            con,
        )
        sorted_result: pd.DataFrame = result.sort_values("image_path").reset_index()

        image_items: List[ImageItem] = [
            ImageItem(
                id=ImageId(sorted_result.iat[index, 1]),
                display_name=ImageName(sorted_result.iat[index, 2]),
                tags=ImageTags(sorted_result.iat[index, 3]),
            )
            for index, _ in enumerate(tqdm.tqdm(sorted_result["image_id"]))
        ]
        print("end:load_index_item_list")
        return image_items

    @cache
    def load_aesthetic_quality_list(self, model_id: ModelId, aesthetic_model_name) -> Dict[ImageId, float]:
        print("start:load_aesthetic_quality_list")
        con: sqlite3.Connection = sqlite3.connect(
            f"{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/sqlite_image_meta.db",
            isolation_level="DEFERRED",
        )
        result: pd.DataFrame = pd.read_sql_query(
            """
            SELECT image_id, aesthetic_quality, pony_aesthetic_quality FROM image_meta
            """,
            con,
        )
        aesthetic_name = "pony_aesthetic_quality" if not aesthetic_model_name == "original" else "aesthetic_quality"
        aesthetic_quality_item: Dict[ImageId, float] = {
            ImageId(id=str(id)): float(q)
            for id, q in tqdm.tqdm(zip(result["image_id"], result[aesthetic_name]))
        }
        print("end:load_aesthetic_quality_list")
        return aesthetic_quality_item

    def load_image_meta_info(self, model_id: ModelId, image_id: ImageId) -> dict[str, str]:
        con: sqlite3.Connection = sqlite3.connect(
            f"{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/sqlite_image_meta.db",
            isolation_level="DEFERRED",
        )
        cur = con.cursor()
        cur.execute(
            "SELECT style_cluster, rating FROM image_meta WHERE image_id = ?",
            (image_id.id,),
        )
        row = cur.fetchone()
        con.close()

        if row is None:
            return {"style_cluster": "", "rating": ""}

        style_cluster, rating = row
        return {
            "style_cluster": style_cluster or "",
            "rating": rating or "",
        }
