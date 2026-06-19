#!/usr/bin/env python
"""
Single-image inference for Qwen3-VL-Embedding through a vLLM
OpenAI-compatible HTTP API.

The vLLM server must be started with ``--runner pooling``.

Examples:
    uv run python qwen3_vl_embed_image_vllm_api.py image.jpg
    uv run python qwen3_vl_embed_image_vllm_api.py image.jpg \
        --warmup 1 --repeat 5 --output embedding.npy
    uv run python qwen3_vl_embed_image_vllm_api.py "C:\\images\\1.png" \
        --api-base http://127.0.0.1:8000/v1

Image preprocessing options such as min_pixels, max_pixels, and max_model_len
belong to the vLLM server process, not this client.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from PIL import Image, UnidentifiedImageError


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION = "Represent the user's input."
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"
SUPPORTED_DIRECT_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Qwen3-VL-Embedding vector for one local image by "
            "calling a vLLM OpenAI-compatible API server."
        )
    )
    parser.add_argument(
        "image_positional",
        nargs="?",
        type=Path,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--image",
        dest="image_option",
        type=Path,
        default=None,
        help="Path to the input image. Overrides the positional path.",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("VLLM_API_BASE", DEFAULT_API_BASE),
        help=(
            "vLLM OpenAI-compatible API base URL. The /v1 suffix is added "
            f"when omitted. Default: {DEFAULT_API_BASE}"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
        help="API key configured on the vLLM server. Default: EMPTY.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL_ID", DEFAULT_MODEL_ID),
        help=(
            "Served model name exposed by vLLM. "
            f"Default: {DEFAULT_MODEL_ID}"
        ),
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="System instruction used when creating the embedding.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=2048,
        help=(
            "Output dimension after Matryoshka truncation and L2 "
            "renormalization. For the 2B model: 64 to 2048. Default: 2048."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of unmeasured API inference requests. Default: 0.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of measured API inference requests. Default: 1.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds. Default: 300.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum automatic HTTP retries by the OpenAI client. Default: 2.",
    )
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip the initial GET /v1/models connectivity check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Supported: .npy or .json.",
    )
    parser.add_argument(
        "--print-values",
        type=int,
        default=16,
        help="Number of leading embedding values to print. Default: 16.",
    )
    parser.add_argument(
        "--print-full",
        action="store_true",
        help="Print the complete embedding vector as JSON.",
    )
    return parser.parse_args()


def normalize_api_base(api_base: str) -> str:
    normalized = api_base.strip().rstrip("/")
    if not normalized:
        raise ValueError("--api-base must not be empty.")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


def image_to_data_url(path: Path) -> tuple[str, dict[str, Any]]:
    """Validate an image and encode it as a Data URL.

    Common web image formats are sent without transcoding. Other Pillow-readable
    formats are converted losslessly to PNG so the API server receives a format
    with broad multimodal-loader support.
    """
    raw = path.read_bytes()

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            width, height = image.size
            image_format = image.format
            detected_mime = Image.MIME.get(image_format or "")

            guessed_mime, _ = mimetypes.guess_type(path.name)
            mime_type = detected_mime or guessed_mime or "application/octet-stream"

            transcoded = mime_type not in SUPPORTED_DIRECT_MIME_TYPES
            if transcoded:
                converted = image.convert("RGB")
                buffer = io.BytesIO()
                converted.save(buffer, format="PNG", optimize=False)
                raw = buffer.getvalue()
                mime_type = "image/png"
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unsupported or damaged image: {path}") from exc

    encoded = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    diagnostics = {
        "width": width,
        "height": height,
        "source_format": image_format,
        "sent_mime_type": mime_type,
        "source_bytes": path.stat().st_size,
        "sent_bytes": len(raw),
        "data_url_chars": len(data_url),
        "transcoded_to_png": transcoded,
    }
    return data_url, diagnostics


def build_messages(
    *,
    image_data_url: str,
    instruction: str,
) -> list[dict[str, Any]]:
    """Build the Qwen3-VL-Embedding message layout used by vLLM."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
                {"type": "text", "text": ""},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
        },
    ]


def create_chat_embedding(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
) -> CreateEmbeddingResponse:
    """Call vLLM's multimodal extension of POST /v1/embeddings."""
    return client.post(
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body={
            "model": model,
            "messages": messages,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        },
    )


def truncate_and_normalize(
    embedding_values: list[float],
    output_dim: int,
) -> np.ndarray:
    raw = np.asarray(embedding_values, dtype=np.float32)
    if raw.ndim != 1:
        raise ValueError(f"Expected a 1-D embedding, got shape {raw.shape}.")
    if raw.size < output_dim:
        raise ValueError(
            f"Server returned {raw.size} dimensions, fewer than "
            f"--output-dim={output_dim}."
        )

    embedding = np.ascontiguousarray(raw[:output_dim])
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid embedding norm: {norm!r}.")
    embedding /= norm
    return embedding


def save_embedding(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        np.save(path, embedding)
    elif suffix == ".json":
        path.write_text(
            json.dumps(embedding.tolist(), ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        raise ValueError(
            f"Unsupported output format: {path.suffix!r}. Use .npy or .json."
        )


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def print_api_error(exc: Exception) -> None:
    if isinstance(exc, APIStatusError):
        print(
            f"API request failed: HTTP {exc.status_code}: {exc.message}",
            file=sys.stderr,
        )
        try:
            print(f"response: {exc.response.text}", file=sys.stderr)
        except Exception:
            pass
    elif isinstance(exc, APITimeoutError):
        print(f"API request timed out: {exc}", file=sys.stderr)
    elif isinstance(exc, APIConnectionError):
        print(f"Could not connect to the vLLM API server: {exc}", file=sys.stderr)
    else:
        print(f"Inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    image_path = args.image_option or args.image_positional or Path("images/1.png")

    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2
    if not 64 <= args.output_dim <= 2048:
        print("--output-dim must be between 64 and 2048 for the 2B model.", file=sys.stderr)
        return 2
    if args.repeat < 1 or args.warmup < 0:
        print("--repeat must be >= 1 and --warmup must be >= 0.", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be greater than 0.", file=sys.stderr)
        return 2
    if args.max_retries < 0:
        print("--max-retries must be >= 0.", file=sys.stderr)
        return 2

    try:
        api_base = normalize_api_base(args.api_base)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print("=== API client ===")
    print(f"API base: {api_base}")
    print(f"model: {args.model}")
    print(f"timeout: {args.timeout:.1f} s")
    print(f"max retries: {args.max_retries}")

    encode_start = time.perf_counter()
    try:
        image_data_url, image_info = image_to_data_url(image_path)
    except (OSError, ValueError) as exc:
        print(f"Image loading failed: {exc}", file=sys.stderr)
        return 1
    encode_seconds = time.perf_counter() - encode_start

    print("\n=== Image ===")
    print(f"path: {image_path.resolve()}")
    print(f"size: {image_info['width']}x{image_info['height']}")
    print(f"source format: {image_info['source_format']}")
    print(f"sent MIME type: {image_info['sent_mime_type']}")
    print(f"source bytes: {format_bytes(image_info['source_bytes'])}")
    print(f"sent bytes: {format_bytes(image_info['sent_bytes'])}")
    print(f"transcoded to PNG: {image_info['transcoded_to_png']}")
    print(f"Data URL preparation: {encode_seconds:.3f} s")

    messages = build_messages(
        image_data_url=image_data_url,
        instruction=args.instruction,
    )
    client = OpenAI(
        base_url=api_base,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    if not args.skip_server_check:
        try:
            model_list = client.models.list()
            served_models = [item.id for item in model_list.data]
            print("\n=== Server ===")
            print(f"served models: {served_models}")
            if served_models and args.model not in served_models:
                print(
                    "Warning: --model is not present in GET /v1/models. "
                    "Use the exact --served-model-name configured on the server.",
                    file=sys.stderr,
                )
        except Exception as exc:
            print_api_error(exc)
            return 1

    try:
        for index in range(args.warmup):
            _ = create_chat_embedding(
                client,
                model=args.model,
                messages=messages,
            )
            print(f"warmup {index + 1}/{args.warmup}: complete")

        elapsed_times: list[float] = []
        response: CreateEmbeddingResponse | None = None
        for index in range(args.repeat):
            started = time.perf_counter()
            response = create_chat_embedding(
                client,
                model=args.model,
                messages=messages,
            )
            elapsed = time.perf_counter() - started
            elapsed_times.append(elapsed)
            print(f"request {index + 1}/{args.repeat}: {elapsed:.3f} s")
    except Exception as exc:
        print_api_error(exc)
        return 1
    finally:
        client.close()

    if response is None or not response.data:
        print("The server returned no embedding data.", file=sys.stderr)
        return 1

    try:
        embedding = truncate_and_normalize(
            response.data[0].embedding,
            args.output_dim,
        )
    except ValueError as exc:
        print(f"Invalid embedding response: {exc}", file=sys.stderr)
        return 1

    print("\n=== Result ===")
    print(f"embedding shape: {embedding.shape}")
    print(f"embedding dtype: {embedding.dtype}")
    print(f"L2 norm: {np.linalg.norm(embedding):.8f}")
    print(f"finite: {bool(np.isfinite(embedding).all())}")
    print(f"request mean: {np.mean(elapsed_times):.3f} s")
    print(f"request min: {np.min(elapsed_times):.3f} s")
    print(f"request max: {np.max(elapsed_times):.3f} s")

    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is not None:
            print(f"prompt tokens: {prompt_tokens}")
        if total_tokens is not None:
            print(f"total tokens: {total_tokens}")

    if args.print_full:
        print("embedding:")
        print(json.dumps(embedding.tolist()))
    else:
        count = max(0, min(args.print_values, embedding.size))
        print(f"first {count} values:")
        print(np.array2string(embedding[:count], precision=8, separator=", "))

    if args.output is not None:
        try:
            save_embedding(args.output, embedding)
        except (OSError, ValueError) as exc:
            print(f"Saving failed: {exc}", file=sys.stderr)
            return 1
        print(f"saved: {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
