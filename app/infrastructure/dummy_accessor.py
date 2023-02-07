
from typing import Any


import numpy as np

from app.domain.domain_object import ImageId, ModelId
from app.application.accessor import Accessor


class DummyAccessor(Accessor):
    """ダミーのAccessor"""


    def load_meta(self, id: ImageId) -> np.ndarray:
        return None # type: ignore
    
    def load_index_file(self, id : ModelId) -> Any:
        return None

