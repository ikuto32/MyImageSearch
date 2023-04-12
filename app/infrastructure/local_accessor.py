
from functools import cache
import json
import sqlite3
from typing import Any, Dict, List

import pathlib

import numpy as np
import pandas as pd
import torch
import faiss
import open_clip

from app.domain.domain_object import ImageId, ImageItem, ImageName, ModelId, Model, Tokenizer
from app.application.accessor import Accessor


class LocalAccessor(Accessor):
    """ローカル上のファイルを対象としたAccessor"""

    def __init__(
        self,
        meta_dir_path: pathlib  # type: ignore
    ) -> None:

        self._meta_dir_path = meta_dir_path
        self._id_to_path: Dict[ImageId, pathlib.Path] = {}

    def load_meta(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        return np.load(f'{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/{image_id}.npy')

    @cache
    def load_index_file(self, model_id: ModelId) -> Any:

        index = faiss.read_index(
            f'{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/{"metafiles.index"}')
        return index

    @cache
    def load_model(self, model_id: ModelId) -> Model:

        model_name, pretrained = model_id.model_name, model_id.pretrained
        return Model(open_clip.create_model_and_transforms(model_name, pretrained=pretrained))

    @cache
    def load_tokenizer(self, model_id: ModelId) -> Tokenizer:
        return Tokenizer(open_clip.get_tokenizer(model_id.model_name))

    @cache
    def load_index_item_list(self, model_id: ModelId) -> List[ImageItem]:
        print("start:load_index_item_list")
        con: sqlite3.Connection = sqlite3.connect(f'{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/sqlite_image_meta.db', isolation_level="DEFERRED")
        result: pd.DataFrame = pd.read_sql_query("""
            SELECT image_id, image_path FROM image_meta
            """, con)
        sorted_result: pd.DataFrame= result.sort_values('image_path').reset_index()
        print("end:load_index_item_list")
        image_item: List[ImageItem] = [ImageItem(id=ImageId(sorted_result.iat[index, 1]), display_name=ImageName(sorted_result.iat[index, 2])) for index, _ in enumerate(sorted_result["image_id"])]
        return image_item

    @cache
    def load_aesthetic_quality_list(self, model_id: ModelId) -> Dict[ImageId, float]:
        print("start:load_aesthetic_quality_list")
        con: sqlite3.Connection = sqlite3.connect(f'{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/sqlite_image_meta.db', isolation_level="DEFERRED")
        result: pd.DataFrame = pd.read_sql_query("""
            SELECT image_id, aesthetic_quality FROM image_meta
            """, con)
        aesthetic_quality_item: Dict[ImageId, float] = {ImageId(id=str(id)):float(q) for id, q in zip(result["image_id"], result["aesthetic_quality"])}
        print("end:load_aesthetic_quality_list")
        return aesthetic_quality_item
