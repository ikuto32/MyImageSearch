#!/usr/bin/env python
"""
Standalone single-image inference for Qwen3-VL-Embedding.

Example:
    uv run python qwen3_vl_embed_image.py "C:\\path\\to\\image.jpg"
    uv run python qwen3_vl_embed_image.py image.jpg --max-pixels 262144 --output embedding.npy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
import transformers
from transformers import AutoProcessor


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION = "Represent the user's input."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Qwen3-VL-Embedding vector for one local image."
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
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Model ID or local model directory. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="System instruction used when creating the embedding.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1_310_720,
        help=(
            "Maximum image pixels passed to Qwen preprocessing. "
            "Use 262144 for a faster 512x512-equivalent test. "
            "Default: 1310720."
        ),
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=4096,
        help="Minimum image pixels. Default: 4096.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum token sequence length. Default: 8192.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=2048,
        help="Output embedding dimension, from 64 to 2048. Default: 2048.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Model dtype. Default: float16.",
    )
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
        help="Attention implementation. Default: sdpa.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of unmeasured warm-up forwards. Default: 0.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of measured forwards. Default: 1.",
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


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping[name]

    if device.type == "cpu" and dtype == torch.float16:
        print(
            "Warning: float16 on CPU is not recommended; using float32 instead.",
            file=sys.stderr,
        )
        return torch.float32

    return dtype


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.3f} GiB"


def build_inputs(
    processor: Any,
    image: Image.Image,
    *,
    instruction: str,
    min_pixels: int,
    max_pixels: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            ],
        },
    ]
    conversations = [conversation]

    prompt = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=False,
    )

    images, video_inputs, video_kwargs = process_vision_info(
        conversations,
        image_patch_size=16,
        return_video_metadata=True,
        return_video_kwargs=True,
    )

    if video_inputs is not None:
        videos, video_metadata = zip(*video_inputs)
        videos = list(videos)
        video_metadata = list(video_metadata)
    else:
        videos = None
        video_metadata = None

    inputs = processor(
        text=prompt,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        truncation=True,
        max_length=max_length,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )

    return dict(inputs)


def move_inputs(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }


def pool_last_valid_token(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "Expected last_hidden_state with shape (batch, sequence, hidden), "
            f"got {tuple(last_hidden_state.shape)}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            "Expected attention_mask with shape (batch, sequence), "
            f"got {tuple(attention_mask.shape)}"
        )

    last_from_end = attention_mask.flip(dims=[1]).argmax(dim=1)
    columns = attention_mask.shape[1] - last_from_end - 1
    rows = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[rows, columns]


def print_input_diagnostics(inputs: dict[str, torch.Tensor]) -> None:
    print("\n=== Input diagnostics ===")
    for key, value in inputs.items():
        if torch.is_tensor(value):
            print(
                f"{key}: shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device}"
            )

    grid = inputs.get("image_grid_thw")
    if torch.is_tensor(grid):
        grid_cpu = grid.detach().to("cpu", dtype=torch.int64)
        visual_tokens = grid_cpu.prod(dim=-1) // 4
        print(f"image_grid_thw: {grid_cpu.tolist()}")
        print(f"estimated merged visual tokens: {visual_tokens.tolist()}")

    input_ids = inputs.get("input_ids")
    if torch.is_tensor(input_ids):
        print(f"sequence length: {input_ids.shape[1]}")


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


def main() -> int:
    args = parse_args()
    args.image = args.image_option or args.image_positional or Path("images/1.png")

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 2

    if not 64 <= args.output_dim <= 2048:
        print("--output-dim must be between 64 and 2048.", file=sys.stderr)
        return 2

    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        print(
            "--max-pixels must be greater than or equal to --min-pixels.",
            file=sys.stderr,
        )
        return 2

    if args.repeat < 1 or args.warmup < 0:
        print("--repeat must be >= 1 and --warmup must be >= 0.", file=sys.stderr)
        return 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)

    print("=== Runtime ===")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"compute capability: {torch.cuda.get_device_capability(device)}")
        print(f"CUDA runtime: {torch.version.cuda}")
    print(f"dtype: {dtype}")
    print(f"attention: {args.attention}")
    print(f"model: {args.model}")

    model_load_start = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(
            args.model,
            padding_side="right",
            trust_remote_code=True,
        )
        model = transformers.Qwen3VLModel.from_pretrained(
            args.model,
            dtype=dtype,
            attn_implementation=args.attention,
            trust_remote_code=True,
        ).to(device)
    except Exception as exc:
        print(
            f"Model loading failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if args.attention == "flash_attention_2":
            print(
                "flash_attention_2 may not be installed or supported in this environment. "
                "Retry with --attention sdpa.",
                file=sys.stderr,
            )
        return 1

    model.eval()
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start

    print("\n=== Model ===")
    print(f"class: {type(model).__name__}")
    print(f"parameter device: {next(model.parameters()).device}")
    print(f"parameter dtype: {next(model.parameters()).dtype}")
    print(
        "root attention:",
        getattr(model.config, "_attn_implementation", None),
    )
    text_config = getattr(model.config, "text_config", None)
    vision_config = getattr(model.config, "vision_config", None)
    if text_config is not None:
        print(
            "text attention:",
            getattr(text_config, "_attn_implementation", None),
        )
    if vision_config is not None:
        print(
            "vision attention:",
            getattr(vision_config, "_attn_implementation", None),
        )
    print(f"model load: {model_load_seconds:.3f} s")

    image_load_start = time.perf_counter()
    try:
        with Image.open(args.image) as source:
            image = source.convert("RGB").copy()
    except Exception as exc:
        print(
            f"Image loading failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    image_load_seconds = time.perf_counter() - image_load_start
    print("\n=== Image ===")
    print(f"path: {args.image.resolve()}")
    print(f"size: {image.width}x{image.height}")
    print(f"image load: {image_load_seconds:.3f} s")
    print(f"min_pixels: {args.min_pixels}")
    print(f"max_pixels: {args.max_pixels}")

    preprocess_start = time.perf_counter()
    try:
        inputs_cpu = build_inputs(
            processor,
            image,
            instruction=args.instruction,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_length=args.max_length,
        )
    except Exception as exc:
        print(
            f"Preprocessing failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    preprocess_seconds = time.perf_counter() - preprocess_start

    transfer_start = time.perf_counter()
    inputs = move_inputs(inputs_cpu, device)
    synchronize(device)
    transfer_seconds = time.perf_counter() - transfer_start

    print_input_diagnostics(inputs)
    print(f"preprocess: {preprocess_seconds:.3f} s")
    print(f"host-to-device: {transfer_seconds:.3f} s")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def run_forward() -> torch.Tensor:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                return_dict=True,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
                hidden = outputs[0]
            if hidden is None:
                raise RuntimeError(
                    f"{type(outputs).__name__} did not return last_hidden_state."
                )

            embedding = pool_last_valid_token(
                hidden,
                inputs["attention_mask"],
            )
            embedding = embedding[..., : args.output_dim].contiguous()
            embedding = F.normalize(embedding.float(), p=2, dim=-1)
            return embedding

    try:
        for index in range(args.warmup):
            _ = run_forward()
            synchronize(device)
            print(f"warmup {index + 1}/{args.warmup}: complete")

        times: list[float] = []
        embedding = None

        for index in range(args.repeat):
            synchronize(device)
            forward_start = time.perf_counter()
            embedding = run_forward()
            synchronize(device)
            elapsed = time.perf_counter() - forward_start
            times.append(elapsed)
            print(f"forward {index + 1}/{args.repeat}: {elapsed:.3f} s")

        assert embedding is not None
    except Exception as exc:
        print(
            f"Forward failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    embedding_np = (
        embedding[0]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )

    print("\n=== Result ===")
    print(f"embedding shape: {embedding_np.shape}")
    print(f"embedding dtype: {embedding_np.dtype}")
    print(f"L2 norm: {np.linalg.norm(embedding_np):.8f}")
    print(f"finite: {bool(np.isfinite(embedding_np).all())}")
    print(f"forward mean: {np.mean(times):.3f} s")
    print(f"forward min: {np.min(times):.3f} s")
    print(f"forward max: {np.max(times):.3f} s")

    if device.type == "cuda":
        print(
            "peak CUDA allocated:",
            format_bytes(torch.cuda.max_memory_allocated(device)),
        )
        print(
            "peak CUDA reserved:",
            format_bytes(torch.cuda.max_memory_reserved(device)),
        )

    if args.print_full:
        print("embedding:")
        print(json.dumps(embedding_np.tolist()))
    else:
        count = max(0, min(args.print_values, embedding_np.size))
        print(f"first {count} values:")
        print(np.array2string(embedding_np[:count], precision=8, separator=", "))

    if args.output is not None:
        try:
            save_embedding(args.output, embedding_np)
        except Exception as exc:
            print(
                f"Saving failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        print(f"saved: {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
