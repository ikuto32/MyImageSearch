import pathlib
from typing import Any

import numpy as np

from app.domain.domain_object import ImageId, ImageItem, ModelId
from app.application.accessor import Accessor
from app.application.embedding_backend import SearchEmbeddingBackend


class DummyAccessor(Accessor):
    """ダミーのAccessor"""

    def load_image_feature(self, model_id: ModelId, id: ImageId) -> np.ndarray:
        return None  # type: ignore

    def load_model(self, model_id: ModelId):
        return None  # type: ignore

    def load_tokenizer(self, model_id: ModelId):
        return None  # type: ignore

    def load_embedding_backend(self, model_id: ModelId) -> SearchEmbeddingBackend:
        return None  # type: ignore

    def load_index_with_metadata(self, model_id: ModelId, aesthetic_model_name: str):
        return None, []

    def load_startup_image_items(
        self,
        model_id: ModelId,
    ) -> tuple[list[ImageItem], dict[ImageId, pathlib.Path]]:
        return [], {}

    def get_mean_meta_vector(self, model_id: ModelId) -> np.ndarray | None:
        return None
