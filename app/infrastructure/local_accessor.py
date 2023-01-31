
from typing import Any

import pathlib

import numpy as np
import torch
import faiss
import open_clip

from app.domain.domain_object import ImageId, ModelId, Model, Tokenizer
from app.application.accessor import Accessor


class LocalAccessor(Accessor):
    """ローカル上のファイルを対象としたAccessor"""


    def __init__(
            self,
            meta_dir_path : pathlib
            ):
        
        self._meta_dir_path = meta_dir_path
        self._id_to_path : dict[ImageId, pathlib.Path] = {}


    def load_meta(self, id: ImageId) -> np.ndarray:
        
        return np.load(f'{self._meta_dir_path}/{id}.npy')


    def load_index_file(self, id: ModelId) -> Any:
        # TODO
        
        index = faiss.read_index(f'{self._meta_dir_path}/{id.model_name}-{id.pretrained}/{"metafiles.index"}')
        return index
    
    
    def load_model(self, id: ModelId) -> list[Model]:
        
        model_name, pretrained = id.model_name, id.pretrained
        return Model(open_clip.create_model_and_transforms(model_name, pretrained=pretrained))
    
    def load_tokenizer(self, id : ModelId) -> Tokenizer:
        return open_clip.get_tokenizer(id.model_name)

