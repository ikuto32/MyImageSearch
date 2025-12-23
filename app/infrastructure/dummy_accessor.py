from typing import Any

import numpy as np

from app.domain.domain_object import ImageId, ModelId
from app.application.accessor import Accessor


class DummyAccessor(Accessor):
    """ダミーのAccessor"""

    def load_meta(self, model_id: ModelId, id: ImageId) -> np.ndarray:
        return None  # type: ignore

    def load_index_file(self, id: ModelId) -> Any:
        return None

    def load_model(self, model_id: ModelId):
        return None  # type: ignore

    def load_tokenizer(self, model_id: ModelId):
        return None  # type: ignore

    def load_index_item_list(self, model_id: ModelId):
        return []

    def load_aesthetic_quality_list(self, model_id: ModelId, aesthetic_model_name):
        return {}

    def load_image_meta_info(self, model_id: ModelId, image_id: ImageId):
        return {"style_cluster": "", "rating": ""}

    def load_style_cluster_list(self, model_id: ModelId):
        return {}

    def load_rating_list(self, model_id: ModelId):
        return {}
