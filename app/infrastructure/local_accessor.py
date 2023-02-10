
from functools import cache
import json
from typing import Any

import pathlib

import numpy as np
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
        self._id_to_path: dict[ImageId, pathlib.Path] = {}

    def load_meta(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        return np.load(f'{self._meta_dir_path}/{model_id.model_name}-{model_id.pretrained}/{image_id}.npy')

    @cache
    def load_index_file(self, id: ModelId) -> Any:

        index = faiss.read_index(
            f'{self._meta_dir_path}/{id.model_name}-{id.pretrained}/{"metafiles.index"}')
        return index

    @cache
    def load_model(self, id: ModelId) -> Model:

        model_name, pretrained = id.model_name, id.pretrained
        return Model(open_clip.create_model_and_transforms(model_name, pretrained=pretrained))

    @cache
    def load_tokenizer(self, id: ModelId) -> Tokenizer:
        return Tokenizer(open_clip.get_tokenizer(id.model_name))

    @cache
    def load_index_item_list(self, id: ModelId) -> list[ImageItem]:
        with open(f'{self._meta_dir_path}/{id.model_name}-{id.pretrained}/{"item_json.json"}', 'r') as f:
            json_dict = json.load(f)
            image_item: list[ImageItem] = [ImageItem(id=ImageId(id), display_name=ImageName(json_dict[id]["path"])) for id in json_dict]
        return image_item
