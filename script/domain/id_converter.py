
from script.domain import Item

class IdConverter:

    _id_to_item = {}
    _path_to_item = {}

    def register(self, item : Item) -> None:

        self._id_to_item[item.id] = item
        self._path_to_item[item.path] = item
        return

    def get_item_from_id(self, id : str):
        return self._id_to_item.get(id)

    def get_item_from_path(self, path : str):
        return self._path_to_item.get(path)