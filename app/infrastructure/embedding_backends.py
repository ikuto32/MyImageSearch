"""検索クエリを埋め込みに変換するモデル。各ライブラリは構築時に初めて読み込む。"""
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from app.application.embedding_backend import SearchEmbeddingBackend, to_float32_2d_array
from app.domain.domain_object import ModelId


class OpenClipEmbeddingBackend(SearchEmbeddingBackend):
    """OpenCLIPの検索エンコーダ。"""

    def __init__(self, model_name: str, pretrained: str) -> None:
        import open_clip

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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
        self.model = transformers.Qwen3VLModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()

    def _move_inputs_to_device(self, inputs):
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, Mapping):
            return {
                key: self._move_inputs_to_device(value)
                for key, value in inputs.items()
            }
        if isinstance(inputs, tuple):
            return tuple(self._move_inputs_to_device(value) for value in inputs)
        if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, bytearray)):
            return [self._move_inputs_to_device(value) for value in inputs]
        return inputs

    def _inputs(self, content: dict, plain: dict, **media):
        """利用者の発話としてチャットテンプレートで整形する。使えないprocessorには素の入力を渡す。"""
        if hasattr(self.processor, "apply_chat_template"):
            try:
                prompt = self.processor.apply_chat_template(
                    [{"role": "user", "content": [content]}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return self.processor(text=[prompt], padding=True, return_tensors="pt", **media)
            except Exception:
                pass
        return self.processor(return_tensors="pt", **plain)

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

    def _encode(self, inputs) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(**self._move_inputs_to_device(inputs), return_dict=True)
            return to_float32_2d_array(self._pool_outputs(outputs))

    def encode_text(self, text: str) -> np.ndarray:
        return self._encode(self._inputs(
            {"type": "text", "text": text}, {"text": [text], "padding": True},
        ))

    def encode_image(self, image) -> np.ndarray:
        return self._encode(self._inputs(
            {"type": "image", "image": image}, {"images": image}, images=[image],
        ))


def _is_qwen_model(model_id: ModelId) -> bool:
    return "qwen" in model_id.model_name.lower() or "qwen" in model_id.pretrained.lower()


def _hf_model_name(model_id: ModelId) -> str:
    if "/" in model_id.model_name or not model_id.pretrained:
        return model_id.model_name
    if "/" in model_id.pretrained:
        return model_id.pretrained
    return model_id.model_name


def create_embedding_backend(model_id: ModelId) -> SearchEmbeddingBackend:
    if _is_qwen_model(model_id):
        return QwenEmbeddingBackend(_hf_model_name(model_id))
    return OpenClipEmbeddingBackend(model_id.model_name, model_id.pretrained)
