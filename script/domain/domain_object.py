

#アプリケーションの設定を示すクラス
class Config:

    def __init__(self, meta_dir: str):
        self._meta_dir = meta_dir

    def get_meta_dir(self):
        return self.meta_dir

#各項目を表すクラス
class Item:


    def __init__(self, id : str, path : str):

        self._id : str = id
        self._path : str = path    

    def get_id(self) -> str:
        return self._id

    def get_path(self) -> str: 
        return self._path




