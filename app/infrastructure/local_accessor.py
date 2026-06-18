from functools import cache
import logging
import sqlite3
from typing import Any, Dict, List, Tuple

import pathlib

import numpy as np
import pandas as pd
import tqdm
import faiss
import open_clip
import torch

from app.domain.domain_object import (
    ImageId,
    ImageItem,
    ImageName,
    ImageTags,
    ModelId,
    Model,
    Tokenizer,
)
from app.application.accessor import Accessor
from app.application.embedding_backend import SearchEmbeddingBackend, to_float32_2d_array
from app.infrastructure.model_metadata import (
    default_model_dir_name,
    has_search_index,
    read_model_metadata,
    safe_model_dir_name,
)


class OpenClipEmbeddingBackend(SearchEmbeddingBackend):
    """既存のOpenCLIP検索エンコーダをSearchEmbeddingBackendとして扱う。"""

    def __init__(self, model_name: str, pretrained: str) -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            return to_float32_2d_array(self.model.encode_text(self.tokenizer([text])))

    def encode_image(self, image) -> np.ndarray:
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to("cpu")
            return to_float32_2d_array(self.model.encode_image(image_tensor))


class QwenEmbeddingBackend(SearchEmbeddingBackend):
    """Hugging FaceのQwen系埋め込みモデルを使う検索エンコーダ。"""

    def __init__(self, model_id: str) -> None:
        import transformers
        from transformers import AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = transformers.Qwen3VLModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _move_inputs_to_device(self, inputs):
        if isinstance(inputs, dict):
            return {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
        return inputs.to(self.device) if torch.is_tensor(inputs) else inputs

    def _structured_text_inputs(self, text: str):
        message = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        if hasattr(self.processor, "apply_chat_template"):
            try:
                prompt = self.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return self.processor(text=[prompt], padding=True, return_tensors="pt")
            except Exception:
                pass
        return self.processor(text=[text], padding=True, return_tensors="pt")

    def _structured_image_inputs(self, image):
        message = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        if hasattr(self.processor, "apply_chat_template"):
            try:
                prompt = self.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
            except Exception:
                pass
        return self.processor(images=image, return_tensors="pt")

    def _pool_outputs(self, outputs):
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(outputs, attr, None)
            if value is not None:
                return value
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        if hidden is None:
            raise RuntimeError("Qwen embedding model did not return embeddings.")
        return hidden.mean(dim=1)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self._move_inputs_to_device(self._structured_text_inputs(text))
            outputs = self.model(**inputs, return_dict=True)
            return to_float32_2d_array(self._pool_outputs(outputs))

    def encode_image(self, image) -> np.ndarray:
        with torch.no_grad():
            inputs = self._move_inputs_to_device(self._structured_image_inputs(image))
            outputs = self.model(**inputs, return_dict=True)
            return to_float32_2d_array(self._pool_outputs(outputs))


def _is_qwen_model(model_id: ModelId) -> bool:
    return "qwen" in model_id.model_name.lower() or "qwen" in model_id.pretrained.lower()


def _hf_model_name(model_id: ModelId) -> str:
    if "/" in model_id.model_name or not model_id.pretrained:
        return model_id.model_name
    if "/" in model_id.pretrained:
        return model_id.pretrained
    return model_id.model_name


class LocalAccessor(Accessor):
    """ローカル上のファイルを対象としたAccessor"""

    def __init__(self, meta_dir_path) -> None:
        self._logger = logging.getLogger(__name__)
        self._meta_dir_path = pathlib.Path(meta_dir_path)
        self._id_to_path: Dict[ImageId, pathlib.Path] = {}

    def _search_model_meta_dir(self, model_id: ModelId) -> pathlib.Path:
        """Return the index directory for a ModelId, including metadata-backed safe names."""
        candidates = [
            self._meta_dir_path / default_model_dir_name(model_id),
            self._meta_dir_path / safe_model_dir_name(model_id.model_name),
        ]
        for candidate in candidates:
            metadata = read_model_metadata(candidate)
            if metadata is not None and (
                metadata.model_name,
                metadata.pretrained,
            ) == (model_id.model_name, model_id.pretrained):
                return candidate
            if metadata is None and candidate.exists():
                return candidate

        if self._meta_dir_path.is_dir():
            for index_dir in self._meta_dir_path.iterdir():
                if not has_search_index(index_dir):
                    continue
                metadata = read_model_metadata(index_dir)
                if metadata is not None and (
                    metadata.model_name,
                    metadata.pretrained,
                ) == (model_id.model_name, model_id.pretrained):
                    return index_dir

        return candidates[0]

    def load_image_feature(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        """meta_dir配下のmodel-pretrained/image_id.npyから画像特徴量を読み込み、毎回np.loadで返す（キャッシュ無し）。"""
        return np.load(
            str(self._search_model_meta_dir(model_id) / f"{image_id}.npy")
        )

    @cache
    def load_model(self, model_id: ModelId) -> Model:
        """モデル名とpretrained設定でopen_clipモデル+変換器を構築しModelに包んで返す（結果は@cacheで共有）。"""

        model_name, pretrained = model_id.model_name, model_id.pretrained
        return Model(
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        )

    @cache
    def load_tokenizer(self, model_id: ModelId) -> Tokenizer:
        """model_id.model_nameに対応するopen_clipトークナイザを生成し、@cacheで同一インスタンスを再利用する。"""
        return Tokenizer(open_clip.get_tokenizer(model_id.model_name))


    @cache
    def load_embedding_backend(self, model_id: ModelId) -> SearchEmbeddingBackend:
        """ModelIdからOpenCLIPまたはQwenの検索埋め込みバックエンドを返す。"""

        if _is_qwen_model(model_id):
            return QwenEmbeddingBackend(_hf_model_name(model_id))
        return OpenClipEmbeddingBackend(model_id.model_name, model_id.pretrained)

    @cache
    def load_index_with_metadata(
        self, model_id: ModelId, aesthetic_model_name: str
    ) -> Tuple[Any, List[ImageItem]]:
        """meta_dir/model-pretrainedのFAISS indexとSQLiteメタを読み込み、画像パス順のImageItemリストとindexをタプルで返す（@cacheで結果共有）。"""
        index = faiss.read_index(
            str(self._search_model_meta_dir(model_id) / "metafiles.index")
        )

        con: sqlite3.Connection = sqlite3.connect(
            str(self._search_model_meta_dir(model_id) / "sqlite_image_meta.db"),
            isolation_level="DEFERRED",
        )
        result: pd.DataFrame = pd.read_sql_query(
            """
            SELECT image_id, image_path, image_tags, aesthetic_quality, pony_aesthetic_quality, style_cluster, rating
            FROM image_meta
            """,
            con,
        )
        con.close()

        aesthetic_name = (
            "pony_aesthetic_quality" if aesthetic_model_name != "original" else "aesthetic_quality"
        )
        sorted_result: pd.DataFrame = result.sort_values("image_path").reset_index(
            drop=True
        )

        image_items: List[ImageItem] = []
        for row in tqdm.tqdm(sorted_result.itertuples(index=False)):
            aesthetic_score = getattr(row, aesthetic_name, None)
            if aesthetic_score is not None and pd.isna(aesthetic_score):
                aesthetic_score = None
            image_items.append(
                ImageItem(
                    id=ImageId(str(row.image_id)),
                    display_name=ImageName(row.image_path),
                    tags=ImageTags(row.image_tags),
                    aesthetic_quality=float(aesthetic_score)
                    if aesthetic_score is not None
                    else None,
                    rating=row.rating or "",
                    style_cluster=row.style_cluster or "",
                )
            )

        return index, image_items

    @cache
    def load_startup_image_items(
        self,
        model_id: ModelId,
    ) -> tuple[list[ImageItem], dict[ImageId, pathlib.Path]]:
        """起動時に必要な画像一覧とImageId->相対パスをSQLiteから構築して返す。"""

        startup_items_by_id: dict[ImageId, ImageItem] = {}
        id_to_path: dict[ImageId, pathlib.Path] = {}
        db_path = (
            self._meta_dir_path
            / f"{model_id.model_name}-{model_id.pretrained}"
            / "sqlite_image_meta.db"
        )

        if not db_path.exists():
            self._logger.warning(
                "起動用のsqlite_image_meta.dbが見つかりません: %s",
                db_path,
            )
            return [], {}

        con: sqlite3.Connection = sqlite3.connect(
            db_path,
            isolation_level="DEFERRED",
        )
        try:
            result: pd.DataFrame = pd.read_sql_query(
                """
                SELECT image_id, image_path, image_tags
                FROM image_meta
                """,
                con,
            )
        finally:
            con.close()

        for row in result.itertuples(index=False):
            image_id = ImageId(str(row.image_id))
            if image_id in startup_items_by_id:
                continue

            image_path = pathlib.Path(str(row.image_path))
            tags = row.image_tags if row.image_tags is not None else ""
            startup_items_by_id[image_id] = ImageItem(
                id=image_id,
                display_name=ImageName(str(image_path)),
                tags=ImageTags(tags),
            )
            id_to_path[image_id] = image_path

        startup_items = sorted(
            startup_items_by_id.values(),
            key=lambda item: item.display_name.name,
        )
        return startup_items, id_to_path

    @cache
    def get_mean_meta_vector(self, model_id: ModelId) -> np.ndarray | None:
        """ModelIdからimage meta全体の平均ベクトルを取得する。"""

        mean_vector = np.load(
            str(self._search_model_meta_dir(model_id) / "metafiles.index.mean.npy")
        )

        return mean_vector
