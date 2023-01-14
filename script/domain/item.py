
# 一つの項目を示すクラス
class Item:


    def __init__(self, id : str, path : str):

        self._id : str = id
        self._path : str = path    

    def get_id(self) -> str:
        return self._id

    def get_path(self) -> str: 
        return self._path