from typing import Any

import numpy as np

from app.domain.domain_object import ImageId, ModelId
from app.application.accessor import Accessor


class DummyAccessor(Accessor):
    """ダミーのAccessor"""

    def load_image_feature(self, model_id: ModelId, id: ImageId) -> np.ndarray:
        return None  # type: ignore

    def load_model(self, model_id: ModelId):
        return None  # type: ignore

    def load_tokenizer(self, model_id: ModelId):
        return None  # type: ignore

    def load_index_with_metadata(self, model_id: ModelId, aesthetic_model_name: str):
        return None, []
