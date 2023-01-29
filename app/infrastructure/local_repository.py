
import pathlib
import uuid
import mimetypes

from app.domain.domain_object import ImageItem, ImageId, Image, ImageName, Model, ModelId, ModelItem
from app.domain.repository import Repository



class LocalRepository(Repository):
    """ローカル上のファイルを対象としたRepository"""


    def __init__(
            self, 
            image_dir_path : pathlib
            ):
        
        self._image_dir_path = image_dir_path
        self._id_to_path : dict[ImageId, pathlib.Path] = {}


    def load_all_image_item(self) -> list[ImageItem]:
        
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
        items : list[ImageItem] = []
        for f in files:

            #ID生成
            id : ImageId = ImageId(str(uuid.uuid4()))
            
            #IDとPathの対応を記録
            self._id_to_path[id] = f

            #Itemに変換
            items.append(ImageItem(id, ImageName(str(f))))

        return items


    def load_all_model_item(self) -> list[ModelItem]:

        #TODO 未実装
        pass
    
    def load_image(self, id: ImageId) -> Image:

        path = self._id_to_path.get(id)
        binary = path.read_bytes()

        # tuple[<Content-Type>, <Encoding>]
        content_type = mimetypes.guess_type(path)[0]

        return Image(binary, content_type)

    def load_model(self, id: ModelId) -> list[Model]:

        #TODO 未実装

        # if device == "auto":
        #     device = "cuda" if torch.cuda.is_available() else "cpu"
        # return open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1])
        
        pass
