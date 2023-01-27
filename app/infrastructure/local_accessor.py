
from typing import Any

import pathlib

import numpy as np
import torch
import faiss
import open_clip

from app.domain.domain_object import ItemId, SearchModelName
from app.application.accessor import Accessor


class LocalAccessor(Accessor):
    """ローカル上のファイルを対象としたAccessor"""


    def __init__(
            self,
            meta_dir_path : pathlib
            ):
        
        self._meta_dir_path = meta_dir_path
        self._id_to_path : dict[ItemId, pathlib.Path] = {}


    def load_meta(self, id: ItemId) -> np.ndarray:
        
        return np.load(f'{self._meta_dir_path}/{id}.npy')


    def load_model(self, model_name: SearchModelName, device: str) -> Any:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1])

    def load_Index_file(self) -> Any:
        return super().loadIndexFile()
