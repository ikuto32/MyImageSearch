
from typing import Any


import numpy as np

from app.domain.domain_object import ItemId
from app.application.accessor import Accessor


class DummyAccessor(Accessor):
    """ダミーのAccessor"""


    def load_meta(self, id: ItemId) -> np.ndarray:
        return None

    def load_model(self, model_name: tuple[str, str], device: str) -> Any:
        return None

    def load_Index_file(self) -> Any:
        return None

