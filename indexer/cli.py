"""検索インデックスを作成・更新するコマンドライン。"""
from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sqlite3
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.infrastructure.index_files import (
    IndexFiles,
    SearchModelMetadata,
    safe_model_dir_name,
    write_model_metadata,
)
from indexer.database import (
    collect_orphan_db_records,
    connect_db,
    delete_orphan_db_records,
    finalize_db,
    find_images_without_embedding,
    image_id_for_path,
    infer_meta_dimension,
    init_db,
)
from indexer.dataset import DirImageIterable, open_rgb, safe_collate
from indexer.embedding import (
    DEFAULT_QWEN_API_BASE,
    DEFAULT_QWEN_INSTRUCTION,
    OpenClipEmbeddingBackend,
    QwenVlEmbeddingBackend,
    SearchEmbeddingBackend,
)
from indexer.faiss_index import build_index
from indexer.features import FeatureExtractor, MetadataModels, recalc_aesthetic_scores
from indexer.image_scan import get_default_image_extensions, scan_image_dir
from indexer.scoring import (
    AESTHETIC_CHECKPOINT_FILENAME,
    AESTHETIC_REPO,
    METADATA_MODEL_NAME,
    METADATA_MODEL_PRETRAINED,
    STYLE_CENTERS_FILENAME,
    STYLE_REPO,
    PonyAestheticScorer,
    StyleCluster,
    ensure_asset_path,
    load_aesthetic_model,
)
from indexer.tagging import ImageTaggingService

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
# 最初の試行だけ、ワーカーから多めに先読みする。
_PREFETCH_FACTOR = 16


def positive_int(value: str) -> int:
    """Argparse type that accepts only positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="画像ディレクトリから検索インデックスを作成・更新する")

    data = parser.add_argument_group("データの場所")
    data.add_argument(
        "--image-dir", default=os.environ.get("MYIMAGESEARCH_IMAGE_DIR") or str(PROJECT_DIR / "images"),
        help="画像ディレクトリ",
    )
    data.add_argument(
        "--meta-dir", default=os.environ.get("MYIMAGESEARCH_META_DIR") or str(PROJECT_DIR / "clip_meta"),
        help="インデックスの保存先。検索モデルごとのディレクトリを作成する",
    )
    data.add_argument(
        "--scan-timeout-sec", type=positive_float, default=120,
        help="進捗のないディレクトリの走査を打ち切るまでの秒数",
    )
    data.add_argument(
        "--delete-orphan-db-records", action="store_true",
        help="画像ディレクトリに存在しない画像の記録をDBから削除する",
    )

    search = parser.add_argument_group("検索モデル")
    search.add_argument("--search-backend", choices=("open_clip", "qwen_vl"), default="qwen_vl")
    search.add_argument("--search-model-name", default="ViT-L-14", help="OpenCLIPのmodel_name（open_clipのみ）")
    search.add_argument("--search-model-pretrained", default="openai", help="OpenCLIPのpretrainedタグ（open_clipのみ）")
    search.add_argument(
        "--search-model-id", default="Qwen/Qwen3-VL-Embedding-2B",
        help="vLLMが提供するモデル名（qwen_vlのみ）",
    )
    search.add_argument(
        "--search-model-out-dim", type=positive_int, default=None,
        help="埋め込みの次元。省略時は保存済みの埋め込み、なければ最初に生成した埋め込みから決める。"
             "QwenのMRLモデルはこの次元に切り詰める",
    )
    search.add_argument("--qwen-api-base", default=os.environ.get("VLLM_API_BASE", DEFAULT_QWEN_API_BASE))
    search.add_argument("--qwen-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    search.add_argument("--qwen-api-timeout", type=positive_float, default=300.0)
    search.add_argument("--qwen-api-max-retries", type=nonnegative_int, default=2)
    search.add_argument("--qwen-api-concurrency", type=positive_int, default=4)
    search.add_argument("--qwen-instruction", default=DEFAULT_QWEN_INSTRUCTION)
    search.add_argument("--qwen-api-skip-server-check", action="store_true")

    metadata = parser.add_argument_group("審美性評価・スタイル・タグ")
    metadata.add_argument(
        "--disable-clip-metadata", action="store_true",
        help="審美性評価とスタイルクラスタを求めず、未設定のまま保存する",
    )
    metadata.add_argument(
        "--aesthetic-model-path", default="./model/aesthetic_rankingv3.pth",
        help="審美性評価(original)の重み。検索モデルが open_clip ViT-L-14 / openai のindexは、"
             "入れ替えた後に --recalc-aesthetic で保存済みの評価を計算し直せる",
    )
    metadata.add_argument("--aesthetic-checkpoint", default=AESTHETIC_REPO, help="ファイルまたはHugging FaceのリポジトリID")
    metadata.add_argument("--style-checkpoint-centers", default=STYLE_REPO, help="ファイルまたはHugging FaceのリポジトリID")
    metadata.add_argument(
        "--use-existing-tags", action="store_true",
        help="タグ付けを行わず、画像の隣の *.tags.json があれば読む",
    )
    metadata.add_argument(
        "--recalc-aesthetic", action="store_true",
        help="画像を読まず、保存済みの埋め込みから審美性評価だけを計算し直してindexを作り直す"
             f"（検索モデルが open_clip の {METADATA_MODEL_NAME} / {METADATA_MODEL_PRETRAINED} の場合のみ）",
    )
    metadata.add_argument("--recalc-aesthetic-batch-size", type=positive_int, default=16384)

    build = parser.add_argument_group("処理とindex")
    build.add_argument("--batch-size", type=positive_int, default=32)
    build.add_argument("--num-workers", type=nonnegative_int, default=8, help="画像を読み込むワーカー数")
    build.add_argument("--nlist", type=positive_int, default=64, help="FAISS IVFのクラスタ数")
    build.add_argument("--pq-m", type=positive_int, default=64, help="FAISS PQの部分ベクトル数（埋め込みの次元ではない）")
    build.add_argument("--bits-per-code", type=positive_int, default=8)

    args = parser.parse_args(argv)
    if args.recalc_aesthetic and not uses_metadata_model_for_search(args):
        # 審美性評価はこのモデルの埋め込みから求める。ほかの検索モデルの埋め込みでは計算できない。
        parser.error(
            "--recalc-aesthetic requires an index whose search model is "
            f"open_clip {METADATA_MODEL_NAME} / {METADATA_MODEL_PRETRAINED} "
            f"(--search-backend open_clip --search-model-name {METADATA_MODEL_NAME} "
            f"--search-model-pretrained {METADATA_MODEL_PRETRAINED})"
        )
    return args


def configure_torch_backends() -> None:
    """Optimize CUDA/cuDNN backends for faster inference."""

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def prepare_index_files(args) -> IndexFiles:
    """検索モデルごとのディレクトリを用意し、検索側がモデルを復元するための情報を書く。"""
    if args.search_backend == "open_clip":
        directory_name = f"{args.search_model_name}-{args.search_model_pretrained}"
        metadata = SearchModelMetadata(args.search_model_name, args.search_model_pretrained, directory_name)
    else:
        directory_name = safe_model_dir_name(args.search_model_id)
        metadata = SearchModelMetadata(args.search_model_id, "", args.search_model_id)
    files = IndexFiles(pathlib.Path(args.meta_dir) / directory_name)
    write_model_metadata(files.directory, metadata)
    return files


def uses_metadata_model_for_search(args) -> bool:
    return (
        args.search_backend == "open_clip"
        and (args.search_model_name, args.search_model_pretrained) == (METADATA_MODEL_NAME, METADATA_MODEL_PRETRAINED)
    )


def search_embedding_model_label(args) -> str:
    if args.search_backend == "open_clip":
        return f"open_clip:{args.search_model_name}:{args.search_model_pretrained}"
    return f"{args.search_backend}:{args.search_model_id}"


def metadata_embedding_model_label(args) -> str | None:
    if args.disable_clip_metadata:
        return None
    return f"open_clip:{METADATA_MODEL_NAME}:{METADATA_MODEL_PRETRAINED}"


def load_search_embedding_backend(args, device: str, dimension: int | None) -> SearchEmbeddingBackend:
    if args.search_backend == "open_clip":
        return OpenClipEmbeddingBackend(args.search_model_name, args.search_model_pretrained, device)
    return QwenVlEmbeddingBackend(
        args.search_model_id,
        dimension,
        args.qwen_api_base,
        args.qwen_api_key,
        args.qwen_api_timeout,
        args.qwen_api_max_retries,
        args.qwen_api_concurrency,
        args.qwen_instruction,
        args.qwen_api_skip_server_check,
    )


def load_metadata_models(args, device: str) -> MetadataModels:
    aesthetic_checkpoint = ensure_asset_path(
        args.aesthetic_checkpoint, AESTHETIC_CHECKPOINT_FILENAME, "aesthetic checkpoint"
    )
    style_centers = ensure_asset_path(
        args.style_checkpoint_centers, STYLE_CENTERS_FILENAME, "style cluster centers"
    )
    return MetadataModels(
        aesthetic=load_aesthetic_model(args.aesthetic_model_path, device),
        pony=PonyAestheticScorer(aesthetic_checkpoint, device),
        style=StyleCluster(style_centers, device),
    )


def has_readable_image(image_dir: str, paths: list[str]) -> bool:
    """開ける画像が1枚でもあるか。開けないファイルは、その理由を表示する。"""
    for path in paths:
        try:
            open_rgb(os.path.join(image_dir, path))
            return True
        except Exception as error:
            print(f"cannot be opened as an image: {path} -> {error}")
    return False


def update_database(args, connection: sqlite3.Connection, device: str) -> int | None:
    """画像ディレクトリを走査し、埋め込みのない画像を処理して保存する。埋め込みの次元を返す。"""
    # 1つのディレクトリには1つの次元だけを保存する。指定がなければ保存済みの次元に合わせる。
    dimension = args.search_model_out_dim or infer_meta_dimension(connection)
    print(f"embedding dimension: {dimension or 'decided by the first embedding'}")

    print(f"scan: {args.image_dir}")
    scan = scan_image_dir(
        args.image_dir, get_default_image_extensions(), scan_timeout_sec=args.scan_timeout_sec,
    )
    for path in scan.unreadable:
        print(f"[scan] could not be read and was skipped: {os.path.join(args.image_dir, path)}")
    image_paths = scan.paths
    image_ids = [image_id_for_path(path) for path in image_paths]

    missing = set(find_images_without_embedding(connection, image_ids, dimension))
    pending = [(path, image_id) for path, image_id in zip(image_paths, image_ids) if image_id in missing]
    orphans = collect_orphan_db_records(connection, image_ids)

    def percent(count: int, total: int) -> str:
        return f"{count}/{total} ({count / total * 100 if total else 0.0:.4f}%)"

    print(f"uncreated images:{percent(len(pending), len(image_paths))}")
    print(f"existing metas:{percent(len(image_paths) - len(pending), len(image_paths))}")
    print(f"orphan db records:{percent(orphans.orphan_count, orphans.total_db_count)}")

    def delete_orphans() -> None:
        if not args.delete_orphan_db_records:
            return
        if not image_paths:
            # 画像ディレクトリを読めなかった場合に、すべての記録を孤立とみなして消さない。
            print("No images were found in the image directory; orphan DB records were not deleted.")
            return
        deletable = orphans
        if scan.unreadable:
            # 読めなかった場所にある画像は、削除されたのか分からないので記録を残す。ルート("")はすべてを残す。
            kept_below = tuple(path and os.path.normcase(path) + os.sep for path in scan.unreadable)
            deletable = orphans._replace(orphan_records=[
                (image_id, image_path) for image_id, image_path in orphans.orphan_records
                if image_path and not (os.path.normcase(image_path) + os.sep).startswith(kept_below)
            ])
            print(
                "Part of the image directory could not be read; "
                f"{orphans.orphan_count - deletable.orphan_count} orphan DB records there were not deleted."
            )
        delete_orphan_db_records(connection, deletable)

    if pending and not has_readable_image(args.image_dir, [path for path, _ in pending]):
        # 壊れたファイルなどは実行のたびに対象になる。それだけのためにモデルを読み込まない。
        print(f"None of the {len(pending)} unprocessed files could be opened as an image.")
        pending = []
    if not pending:
        # 処理する画像がなければ、モデルの読み込みも省く。
        print("No new images to process.")
        delete_orphans()
        return dimension

    print("load_model")
    search_model = load_search_embedding_backend(args, device, dimension)
    metadata_models = metadata_embedding = None
    if not args.disable_clip_metadata:
        metadata_models = load_metadata_models(args, device)
        # 検索モデルがメタデータ用のモデルと同じなら、1回の推論の結果を両方に使う。
        if not uses_metadata_model_for_search(args):
            metadata_embedding = OpenClipEmbeddingBackend(METADATA_MODEL_NAME, METADATA_MODEL_PRETRAINED, device)
    tagging_service = None if args.use_existing_tags else ImageTaggingService()

    extractor = FeatureExtractor(
        connection,
        device=device,
        search_model=search_model,
        search_model_label=search_embedding_model_label(args),
        metadata_models=metadata_models,
        metadata_embedding=metadata_embedding,
        metadata_model_label=metadata_embedding_model_label(args),
        tagging_service=tagging_service,
        image_dir=args.image_dir,
        dimension=dimension,
    )
    extract_with_retries(
        extractor,
        [path for path, _ in pending],
        [image_id for _, image_id in pending],
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=search_model.preprocess,
        metadata_transform=None if metadata_embedding is None else metadata_embedding.preprocess,
        tag_sizes=None if tagging_service is None else tagging_service.input_sizes,
    )
    # モデルの読み込みなどで中断した場合に、古いindexが参照する記録を消さないよう最後に行う。
    delete_orphans()
    return extractor.dimension


def extract_with_retries(
    extractor: FeatureExtractor,
    image_paths: list[str],
    image_ids: list[str],
    *,
    image_dir: str,
    batch_size: int,
    num_workers: int,
    transform,
    metadata_transform,
    tag_sizes: tuple[int, int] | None,
) -> None:
    """画像を処理する。DataLoaderが失敗した場合は、未処理の画像だけをワーカーを減らして再試行する。"""
    seen = np.zeros(len(image_paths), dtype=bool)
    attempt = 0
    while True:
        remaining = np.flatnonzero(~seen)
        if remaining.size == 0:
            return
        workers = max(0, num_workers - 2 * attempt)
        first_attempt = attempt == 0
        loader_options = {"prefetch_factor": _PREFETCH_FACTOR} if workers > 0 and first_attempt else {}
        loader = DataLoader(
            DirImageIterable(
                image_dir,
                [image_paths[index] for index in remaining],
                transform=transform,
                tag_sizes=tag_sizes,
                metadata_transform=metadata_transform,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=first_attempt,
            collate_fn=safe_collate,
            **loader_options,
        )
        print(f"try {attempt}")
        seen_now = np.zeros(len(remaining), dtype=bool)
        try:
            extractor.run(
                loader,
                [image_paths[index] for index in remaining],
                [image_ids[index] for index in remaining],
                seen_now,
                batch_size=batch_size,
            )
            # 読めなかった画像はデータセットが飛ばしている。次回の実行で再び対象になる。
            return
        except Exception:
            traceback.print_exc()
        if seen_now.any():
            seen[remaining[seen_now]] = True
        else:
            # 同じ画像でワーカーが落ち続けないよう、進まなかった場合は先頭の1バッチを飛ばす。
            seen[remaining[:batch_size]] = True
        attempt += 1
        del loader
        gc.collect()
        torch.cuda.empty_cache()


def recalc_database(args, connection: sqlite3.Connection, device: str) -> int:
    """保存済みの埋め込みから審美性評価だけを計算し直す。埋め込みの次元を返す。"""
    print("[recalc_aesthetic] mode: skipping image scan / CLIP inference / tagging")
    dimension = args.search_model_out_dim or infer_meta_dimension(connection)
    if dimension is None:
        raise ValueError("Embedding dimension is unknown because no embeddings are stored.")
    aesthetic_checkpoint = ensure_asset_path(
        args.aesthetic_checkpoint, AESTHETIC_CHECKPOINT_FILENAME, "aesthetic checkpoint"
    )
    recalc_aesthetic_scores(
        connection,
        load_aesthetic_model(args.aesthetic_model_path, device),
        PonyAestheticScorer(aesthetic_checkpoint, device),
        device,
        dimension,
        batch_size=args.recalc_aesthetic_batch_size,
        num_workers=args.num_workers,
    )
    return dimension


def main(argv=None) -> None:
    args = parse_arguments(argv)
    # ファイル名には、出力先の文字コードで表せない文字が含まれることがある。その表示で処理を止めない。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    configure_torch_backends()
    files = prepare_index_files(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    connection = connect_db(files.directory)
    try:
        init_db(connection)
        if args.recalc_aesthetic:
            dimension = recalc_database(args, connection, device)
        else:
            dimension = update_database(args, connection, device)
        finalize_db(connection)
    finally:
        print("close db connection")
        connection.close()
    gc.collect()
    torch.cuda.empty_cache()

    if dimension is None:
        raise ValueError("Embedding dimension is unknown because no embeddings were generated or stored.")
    build_index(files, args.nlist, args.pq_m, args.bits_per_code, dimension)
    if args.recalc_aesthetic:
        print("[recalc_aesthetic] done.")
