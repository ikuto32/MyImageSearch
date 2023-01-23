
import pathlib

import numpy as np

from app.domain.domain_object import Item, ItemId
from app.domain.repository import Repository



class LocalRepository(Repository):
    "ローカル上のファイルを対象としたRepository"


    def __init__(
            self, 
            image_dir_path : pathlib,
            meta_dir_path : pathlib
            ):
        
        self._image_dir_path = image_dir_path
        self._meta_dir_path = meta_dir_path



    def load_items(self) -> list[Item]:
        
        #一覧取得
        files = []
        files.extend(self._image_dir_path.glob(f"**/*.png"))
        files.extend(self._image_dir_path.glob(f"**/*.jpg"))
        files.extend(self._image_dir_path.glob(f"**/*.gif"))
        files.extend(self._image_dir_path.glob(f"**/*.webp"))

        #相対パスに変換
        files = map(lambda f: f.relative_to(self._image_dir_path), files)

        #辞書順に並び替え
        files = sorted(files)

        #Itemに変換
        items = map(lambda f: Item(ItemId(str(f), str(f))), files)

        return items

    
    def load_image_bytes(self, id: ItemId) -> bytes:

        #このリポジトリを使用している場合、IDは、パスとして扱えるはず
        path = pathlib.Path(id.get_id())
        return path.read_bytes()
        

    def load_meta(self, id: ItemId) -> np.ndarray:
        
        #このリポジトリを使用している場合、IDは、パスとして扱えるはず
        path = id.get_id()
        
        return np.load(f'{self._meta_dir_path}/{path}.npy')


