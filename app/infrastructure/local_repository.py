
from typing import Any

import pathlib
import uuid

import numpy as np
import torch
import faiss
import open_clip

from app.domain.domain_object import Item, ItemId, ItemName
from app.domain.repository import Repository



class LocalRepository(Repository):
    """ローカル上のファイルを対象としたRepository"""


    def __init__(
            self, 
            image_dir_path : pathlib,
            meta_dir_path : pathlib
            ):
        
        self._image_dir_path = image_dir_path
        self._meta_dir_path = meta_dir_path
        
        self._id_to_path : dict[ItemId, pathlib.Path] = {}


    def load_items(self) -> list[Item]:
        
        #一覧取得
        files : list[pathlib.Path] = []
        files.extend(self._image_dir_path.glob("**/*.png"))
        files.extend(self._image_dir_path.glob("**/*.jpg"))
        files.extend(self._image_dir_path.glob("**/*.gif"))
        files.extend(self._image_dir_path.glob("**/*.webp"))

        #相対パスに変換
        files = map(lambda f: f.relative_to(self._image_dir_path), files)

        #辞書順に並び替え
        files = sorted(files)

        #PathからIDを決定し、Itemに変換
        items : list[Item] = []
        for f in files:

            #ID生成
            id : ItemId = ItemId(str(uuid.uuid4()))
            
            #IDとPathの対応を記録
            self._id_to_path[id] = f

            #Itemに変換
            items.append(Item(id, ItemName(str(f))))

        return items

    
    def load_image_bytes(self, id: ItemId) -> bytes:

        path = self._id_to_path.get(id)
        return path.read_bytes()
        

    def load_meta(self, id: ItemId) -> np.ndarray:
        
        return np.load(f'{self._meta_dir_path}/{id}.npy')

    def load_model(self, model_name: tuple[str, str], device: str) -> Any:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1])

    def load_Index_file(self) -> Any:
        return super().loadIndexFile()