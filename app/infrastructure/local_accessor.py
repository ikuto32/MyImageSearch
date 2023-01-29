
from typing import Any

import pathlib

import numpy as np
import torch
import faiss
import open_clip

from app.domain.domain_object import ImageId, ModelId
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

        # TODO 未実装
        pass
