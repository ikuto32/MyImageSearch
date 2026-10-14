"""検索用の画像埋め込みを生成するバックエンド。"""
from __future__ import annotations

import base64
import concurrent.futures
import io
import threading
from typing import Any
from urllib.parse import urlparse

import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from PIL import Image
import torch

DEFAULT_QWEN_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_QWEN_INSTRUCTION = "Represent the user's input."


class SearchEmbeddingBackend:
    """検索用の画像埋め込みバックエンド共通インターフェース。"""

    @property
    def preprocess(self):
        """DataLoaderのワーカーで画像1枚に適用する前処理。"""
        raise NotImplementedError

    def encode_image_with_internal(self, image_inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """前処理済みのバッチから(射影前の内部特徴, 検索に使う特徴)を返す。"""
        raise NotImplementedError


class OpenClipEmbeddingBackend(SearchEmbeddingBackend):
    """OpenCLIP ベースの検索エンコーダ。"""

    def __init__(self, model_name: str, pretrained: str, device: str):
        import open_clip

        self.device = device
        self.model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            jit=False,
        )
        self.model.eval()

    @property
    def preprocess(self):
        return self._preprocess

    @torch.no_grad()
    def encode_image_with_internal(self, pixel_values: torch.Tensor):
        visual = self.model.visual
        x = visual._embeds(pixel_values)
        x = visual.transformer(x)
        pooled, _tokens = visual._pool(x)

        proj = getattr(visual, "proj", None)
        projected_features = pooled @ proj if proj is not None else pooled
        return pooled, projected_features


def normalize_qwen_api_base(api_base: str) -> str:
    """Normalize a vLLM OpenAI-compatible API base URL so it ends in /v1."""

    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("Qwen API base URL must not be empty.")
    normalized = api_base.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Invalid Qwen API base URL: {api_base!r}. Expected an http(s) URL such as {DEFAULT_QWEN_API_BASE}."
        )
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def pil_image_to_data_url(image: Image.Image) -> str:
    """Convert a PIL image to a PNG data URL for the vLLM OpenAI API."""

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenVlImagePreprocess:
    """Picklable image transform for Qwen VL search inputs.

    The API server performs the model's preprocessing. DataLoader workers only
    return lightweight PIL images.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.copy()


class QwenVlEmbeddingBackend(SearchEmbeddingBackend):
    """Qwen3-VL-Embedding backend using vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        model_id: str,
        output_dim: int | None = None,
        api_base: str = DEFAULT_QWEN_API_BASE,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        max_retries: int = 2,
        max_concurrency: int = 4,
        instruction: str = DEFAULT_QWEN_INSTRUCTION,
        skip_server_check: bool = False,
    ):
        if timeout <= 0:
            raise ValueError("qwen_api_timeout must be a positive number.")
        if max_retries < 0:
            raise ValueError("qwen_api_max_retries must be 0 or greater.")
        if max_concurrency < 1:
            raise ValueError("qwen_api_concurrency must be 1 or greater.")
        if not instruction:
            raise ValueError("qwen_instruction must not be empty.")

        self.model_id = model_id
        self.output_dim = output_dim
        self.api_base = normalize_qwen_api_base(api_base)
        self.api_key = api_key or "EMPTY"
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.max_concurrency = int(max_concurrency)
        self.instruction = instruction
        self._thread_local = threading.local()

        self.print_qwen_api_config()
        if not skip_server_check:
            self._check_server()

    def print_qwen_api_config(self) -> None:
        print("Qwen backend: vLLM OpenAI-compatible API")
        print(f"API base: {self.api_base}")
        print(f"served model: {self.model_id}")
        print(f"output dimension: {self.output_dim}")
        print(f"timeout: {self.timeout}")
        print(f"max retries: {self.max_retries}")
        print(f"API concurrency: {self.max_concurrency}")
        print(f"instruction: {self.instruction}")
        print(f"API key configured: {bool(self.api_key)}")

    @property
    def preprocess(self):
        return QwenVlImagePreprocess()

    def _get_client(self) -> OpenAI:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            self._thread_local.client = client
        return client

    def _check_server(self) -> None:
        try:
            models = self._get_client().models.list()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise self._runtime_api_error(
                "Could not connect to the Qwen vLLM server. Ensure the vLLM OpenAI-compatible server is running",
                exc,
            ) from exc
        model_ids = [getattr(model, "id", None) for model in getattr(models, "data", [])]
        model_ids = [model_id for model_id in model_ids if model_id]
        if self.model_id not in model_ids:
            available = ", ".join(model_ids) if model_ids else "<none>"
            raise RuntimeError(
                f"Qwen model '{self.model_id}' was not returned by {self.api_base}/models. "
                f"Available models: {available}. If vLLM was started with --served-model-name, "
                "set --search-model-id to that served name."
            )

    def _runtime_api_error(self, message: str, exc: Exception, *, index: int | None = None) -> RuntimeError:
        parts = [f"{message}; API base: {self.api_base}"]
        if index is not None:
            parts.append(f"batch index: {index}")
        if isinstance(exc, APIStatusError):
            parts.append(f"HTTP status: {exc.status_code}")
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    parts.append(f"response body: {response.text}")
                except Exception:
                    pass
        else:
            parts.append(f"error: {exc}")
        return RuntimeError("; ".join(parts))

    def _image_messages(self, image_data_url: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": ""},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

    def _embedding_body(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "messages": messages,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }

    def _post_embedding(self, messages: list[dict[str, Any]], *, index: int | None = None) -> np.ndarray:
        try:
            response = self._get_client().post(
                "/embeddings",
                cast_to=CreateEmbeddingResponse,
                body=self._embedding_body(messages),
            )
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise self._runtime_api_error("Qwen embedding API request failed", exc, index=index) from exc
        return self._validate_embedding_response(response, index=index)

    def _validate_embedding_response(self, response: Any, *, index: int | None = None) -> np.ndarray:
        prefix = f"Qwen embedding response for batch index {index}" if index is not None else "Qwen embedding response"
        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError(f"{prefix} did not contain any data.")
        embedding = getattr(data[0], "embedding", None)
        if embedding is None:
            raise RuntimeError(f"{prefix} did not contain data[0].embedding.")
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1:
            raise RuntimeError(f"{prefix} embedding must be a 1-dimensional numeric array; got shape {array.shape}.")
        if array.size == 0:
            raise RuntimeError(f"{prefix} embedding must not be empty.")
        if not np.isfinite(array).all():
            raise RuntimeError(f"{prefix} embedding contains NaN or Inf values.")
        if self.output_dim is not None:
            if array.shape[0] < self.output_dim:
                raise RuntimeError(
                    f"{prefix} embedding dimension {array.shape[0]} is smaller than requested output_dim {self.output_dim}."
                )
            array = array[: self.output_dim]
        return array

    def _post_image_embedding(self, item: tuple[int, Image.Image]) -> np.ndarray:
        index, image = item
        return self._post_embedding(self._image_messages(pil_image_to_data_url(image)), index=index)

    def encode_image_with_internal(self, images: list[Image.Image]):
        if not isinstance(images, list) or not images or not all(isinstance(image, Image.Image) for image in images):
            raise TypeError(
                f"Qwen vLLM backend expected a non-empty list of PIL images. Got {type(images).__name__}: {images!r}"
            )
        # 結果は入力の順序で返る。画像のPNG化もワーカースレッドで行う。
        max_workers = min(self.max_concurrency, len(images))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(self._post_image_embedding, enumerate(images)))
        features = torch.from_numpy(np.stack(embeddings))
        return features, features
