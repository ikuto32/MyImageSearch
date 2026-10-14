import argparse
import os
import pathlib
from time import perf_counter

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
IMAGE_DIR_VARIABLE = "MYIMAGESEARCH_IMAGE_DIR"
META_DIR_VARIABLE = "MYIMAGESEARCH_META_DIR"
PREFERRED_STARTUP_MODEL = ("ViT-L-14", "openai")


def _data_directory(explicit: pathlib.Path | None, local: bool, variable: str, name: str) -> pathlib.Path:
    """明示した引数 → --local → 環境変数 → プロジェクト内、の順で保存先を決める。"""
    if explicit is not None:
        return explicit
    if not local and os.environ.get(variable):
        return pathlib.Path(os.environ[variable])
    return PROJECT_DIR / name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ローカル画像を検索するWebアプリ")
    parser.add_argument("--local", action="store_true", help="プロジェクト内のimages/clip_metaで軽量実行")
    parser.add_argument("--image-dir", type=pathlib.Path)
    parser.add_argument("--meta-dir", type=pathlib.Path)
    parser.add_argument("--host", default="127.0.0.1", help="待受アドレス（このPCのみは127.0.0.1、LAN共有は0.0.0.0）")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="軽量検証用に起動時のモデル・インデックス読込を省略（既定では読み込む。大規模データは時間・メモリを使用）",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    args.image_dir = _data_directory(args.image_dir, args.local, IMAGE_DIR_VARIABLE, "images")
    args.meta_dir = _data_directory(args.meta_dir, args.local, META_DIR_VARIABLE, "clip_meta")
    return args


def select_startup_model(models):
    if not models:
        raise ValueError("検索モデルが見つかりません。--meta-dir の場所とインデックスを確認してください。")
    return next(
        (item.id for item in models if (item.id.model_name, item.id.pretrained) == PREFERRED_STARTUP_MODEL),
        models[0].id,
    )


def main(argv=None):
    args = parse_args(argv)
    # 引数・ヘルプの確認にはML依存やネットワークアクセスを必要としない。
    from app.application.usecase import Usecase
    from app.infrastructure.local_accessor import LocalAccessor
    from app.infrastructure.local_repository import LocalRepository
    from app.infrastructure.model_registry import ModelRegistry
    from app.presentation.controller import start_app

    registry = ModelRegistry(args.meta_dir)
    try:
        startup_model_id = select_startup_model(registry.items())
    except ValueError as error:
        raise SystemExit(f"{error}（--meta-dir: {args.meta_dir}）") from None
    usecase = Usecase(LocalRepository(args.image_dir, registry), LocalAccessor(registry))

    if not args.skip_warmup:
        started = perf_counter()
        print(f"起動準備: {startup_model_id.model_name} のモデルと検索インデックスを読み込んでいます。", flush=True)
        usecase.warmup_search(startup_model_id)
        # Prepare category totals once before accepting users, so filtering does
        # not trigger a full-catalog aggregation during the first interaction.
        print("起動準備: 画像区分ごとの件数を集計しています。", flush=True)
        usecase.get_catalog_count(startup_model_id)
        usecase.get_catalog_count(startup_model_id, ["general"])
        print(f"起動準備完了（{perf_counter() - started:.1f}秒）。ブラウザから接続できます。", flush=True)
    start_app(usecase, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
