#!/usr/bin/env python
"""Embed one local image through a vLLM OpenAI-compatible API, as create_index.py does.

Use it to check the connection, the served model name and the request latency
before indexing a whole directory. The vLLM server must be started with
``--runner pooling``. Image preprocessing options such as min_pixels and
max_pixels belong to the vLLM server process, not this client.

Examples:
    uv run python scripts/qwen_embed_image.py image.jpg
    uv run python scripts/qwen_embed_image.py image.jpg --warmup 1 --repeat 5 --output .cache/embedding.npy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    from indexer.embedding import DEFAULT_QWEN_API_BASE, DEFAULT_QWEN_INSTRUCTION

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--api-base", default=os.environ.get("VLLM_API_BASE", DEFAULT_QWEN_API_BASE),
                        help="API base URL. The /v1 suffix is added when omitted.")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B"),
                        help="Served model name exposed by vLLM.")
    parser.add_argument("--instruction", default=DEFAULT_QWEN_INSTRUCTION)
    parser.add_argument("--output-dim", type=int, default=None,
                        help="Truncate the Matryoshka embedding to this dimension. Default: the model's dimension.")
    parser.add_argument("--warmup", type=int, default=0, help="Number of unmeasured requests.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of measured requests.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--skip-server-check", action="store_true", help="Skip the initial GET /v1/models check.")
    parser.add_argument("--output", type=Path, default=None, help="Save the embedding as .npy or .json.")
    parser.add_argument("--print-values", type=int, default=16, help="Number of leading values to print.")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be >= 1 and --warmup must be >= 0.")
    if args.output is not None and args.output.suffix.lower() not in (".npy", ".json"):
        parser.error("--output must end in .npy or .json.")
    return args


def main() -> int:
    args = parse_args()
    import numpy as np
    from PIL import UnidentifiedImageError

    from indexer.dataset import open_rgb
    from indexer.embedding import QwenVlEmbeddingBackend

    try:
        image = open_rgb(str(args.image))
    except (OSError, UnidentifiedImageError) as error:
        print(f"Image loading failed: {args.image}: {error}", file=sys.stderr)
        return 2
    print(f"image: {args.image.resolve()} ({image.width}x{image.height})")

    try:
        backend = QwenVlEmbeddingBackend(
            args.model, args.output_dim, args.api_base, args.api_key, args.timeout,
            args.max_retries, 1, args.instruction, args.skip_server_check,
        )
        for index in range(args.warmup):
            backend.encode_image_with_internal([image])
            print(f"warmup {index + 1}/{args.warmup}: complete")
        elapsed_times = []
        for index in range(args.repeat):
            started = time.perf_counter()
            _internal, features = backend.encode_image_with_internal([image])
            elapsed_times.append(time.perf_counter() - started)
            print(f"request {index + 1}/{args.repeat}: {elapsed_times[-1]:.3f} s")
    except (RuntimeError, ValueError) as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1

    # The index stores L2-normalized embeddings.
    embedding = features[0].numpy()
    embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
    print(f"embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"request mean/min/max: {np.mean(elapsed_times):.3f} / {np.min(elapsed_times):.3f} / {np.max(elapsed_times):.3f} s")
    count = max(0, min(args.print_values, embedding.size))
    print(f"first {count} values:")
    print(np.array2string(embedding[:count], precision=8, separator=", "))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".npy":
            np.save(args.output, embedding)
        else:
            args.output.write_text(json.dumps(embedding.tolist()), encoding="utf-8")
        print(f"saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
