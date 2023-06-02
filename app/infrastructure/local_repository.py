
from functools import cache
import hashlib
import itertools
import pathlib
import mimetypes
from typing import List

import torch

from app.domain.domain_object import ImageItem, ImageId, Image, ImageName, Model, ModelId, ModelItem, ModelName
from app.domain.repository import Repository

import open_clip


class LocalRepository(Repository):
    """ローカル上のファイルを対象としたRepository"""


    def __init__(
            self, 
            image_dir_path: pathlib.Path
            ) -> None:
        
        self._image_dir_path: pathlib.Path  = image_dir_path
        self._id_to_path : dict[ImageId, pathlib.Path] = {}

    @cache
    def load_all_image_item(self) -> List[ImageItem]:
        # Define a function to create ImageItem from a file
        def create_image_item(file: pathlib.Path) -> ImageItem:
            relative_file = file.relative_to(self._image_dir_path)
            id = ImageId(hashlib.sha256(str(relative_file).encode()).hexdigest())
            self._id_to_path[id] = relative_file
            return ImageItem(id, ImageName(str(relative_file)))

        # Get all files with specified extensions
        extensions = ["png", "jpg", "gif", "webp"]
        files = itertools.chain.from_iterable(self._image_dir_path.glob(f'**/*.{e}') for e in extensions)

        # Convert files to ImageItems, sort, and return
        items = list(map(create_image_item, files))
        items.sort(key=lambda item: item.display_name.name)

        return items

    @cache
    def load_all_model_item(self) -> list[ModelItem]:

        items: list[ModelItem] = [ModelItem(ModelId(model_name, dataset), ModelName(f"{model_name}-{dataset}")) for (model_name, dataset) in open_clip.list_pretrained()]
        return items
    
    @cache
    def load_image(self, id: ImageId) -> Image:

        path: pathlib.Path = pathlib.Path(f"{self._image_dir_path}/{self._id_to_path.get(id)}")
        binary: bytes = path.read_bytes()

        # tuple[<Content-Type>, <Encoding>]
        content_type = mimetypes.guess_type(path)[0]

        return Image(binary, content_type)# type: ignore
