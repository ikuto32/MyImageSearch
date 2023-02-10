
import hashlib
import itertools
import pathlib
import mimetypes

import torch

from app.domain.domain_object import ImageItem, ImageId, Image, ImageName, Model, ModelId, ModelItem, ModelName
from app.domain.repository import Repository

import open_clip


class LocalRepository(Repository):
    """ローカル上のファイルを対象としたRepository"""


    def __init__(
            self, 
<<<<<<< Updated upstream
            image_dir_path
        ) -> None:
        
        self._image_dir_path: pathlib.Path = pathlib.Path(image_dir_path)
=======
            image_dir_path: pathlib.Path
            ) -> None:
        
        self._image_dir_path: pathlib.Path  = image_dir_path
>>>>>>> Stashed changes
        self._id_to_path : dict[ImageId, pathlib.Path] = {}


    def load_all_image_item(self) -> list[ImageItem]:
        
        #一覧取得
        files : list[pathlib.Path] = []
        extensions: list[str] = ["png", "jpg", "gif", "webp"]
  
        files = itertools.chain.from_iterable(self._image_dir_path.glob(f'**/*.{e}') for e in extensions)
        #相対パスに変換
        files = list(map(lambda f: f.relative_to(self._image_dir_path), files))

        #辞書順に並び替え
        files = sorted(files)

        #PathからIDを決定し、Itemに変換
        items : list[ImageItem] = []
        for f in files:

            #ID生成
            id : ImageId = ImageId(hashlib.sha256(str(f).encode()).hexdigest())
            
            #IDとPathの対応を記録
            self._id_to_path[id] = f

            #Itemに変換
            items.append(ImageItem(id, ImageName(str(f))))

        return items


    def load_all_model_item(self) -> list[ModelItem]:

<<<<<<< Updated upstream
        items: list[ModelItem] = [ModelItem(ModelId(model_name, dataset), ModelName(f"{model_name}-{dataset}")) for (model_name, dataset) in open_clip.list_pretrained()]
=======
        items: list[ModelItem] = [ModelItem(ModelId(model_name, dataset), ModelName( f"{model_name}-{dataset}")) for (model_name, dataset) in open_clip.list_pretrained()]
>>>>>>> Stashed changes
        return items
    
    def load_image(self, id: ImageId) -> Image:

        path: pathlib.Path = pathlib.Path(f"{self._image_dir_path}/{self._id_to_path.get(id)}")
        binary: bytes = path.read_bytes()

        # tuple[<Content-Type>, <Encoding>]
<<<<<<< Updated upstream
        content_type: str = mimetypes.guess_type(path)[0]# type: ignore
        return Image(binary, content_type)

=======
        content_type = mimetypes.guess_type(path)[0]

        return Image(binary, content_type)# type: ignore
>>>>>>> Stashed changes
