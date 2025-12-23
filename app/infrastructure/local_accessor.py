from functools import cache
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

        self._meta_dir_path = meta_dir_path
        self._id_to_path: Dict[ImageId, pathlib.Path] = {}

    def load_image_feature(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        return np.load(
            f"{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/{image_id}.npy"
        )

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
    def load_index_with_metadata(
        self, model_id: ModelId, aesthetic_model_name: str
    ) -> Tuple[Any, List[ImageItem]]:
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
