
from script.domain.domain_object import Config
from script.domain.id_converter import IdConverter

_registory = IdConverter()

def load(config : Config):
    
    #画像の関連情報の読み込みと対応付け
    pass


def get_item_from_id(id : str):
    return _id_to_item.get(id)

def get_item_from_path(path : str):
    return _path_to_item.get(path)
