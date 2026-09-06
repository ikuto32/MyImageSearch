import argparse
import os
import pathlib
from time import perf_counter

PROJECT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ローカル画像を検索するWebアプリ")
    parser.add_argument("--local", action="store_true", help="プロジェクト内のimages/clip_metaで軽量実行")
    parser.add_argument("--image-dir", type=pathlib.Path)
    parser.add_argument("--meta-dir", type=pathlib.Path)
    parser.add_argument("--host", default="127.0.0.1", help="待受アドレス（このPCのみは127.0.0.1、LAN共有は0.0.0.0）")
    parser.add_argument("--port", type=int, default=80)
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup", dest="skip_warmup", action="store_false", help="起動時に検索モデルとインデックスを読み込む（大規模データは時間・メモリを使用）")
    warmup.add_argument("--skip-warmup", dest="skip_warmup", action="store_true", help="軽量検証用に起動時のモデル・インデックス読込を省略")
    parser.set_defaults(skip_warmup=False)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    args.image_dir = args.image_dir or (
        PROJECT_DIR / "images" if args.local else
        pathlib.Path(os.environ.get("MYIMAGESEARCH_IMAGE_DIR") or PROJECT_DIR / "images")
    )
    args.meta_dir = args.meta_dir or (
        PROJECT_DIR / "clip_meta" if args.local else
        pathlib.Path(os.environ.get("MYIMAGESEARCH_META_DIR") or PROJECT_DIR / "clip_meta")
    )
    return args


def main(argv=None):
    args = parse_args(argv)
    # 引数・ヘルプの確認にはML依存やネットワークアクセスを必要としない。
    from app.infrastructure.local_accessor import LocalAccessor
    from app.infrastructure.local_repository import LocalRepository
    from app.presentation.controller import start_app
    from app.application import usecase

    accessor = LocalAccessor(args.meta_dir)

    repository = LocalRepository(args.image_dir, args.meta_dir)
    startup_model_id = select_startup_model(repository.load_all_model_item())

    in_usecase: usecase.Usecase = usecase.Usecase(
        repository, accessor, startup_model_id
    )
    if not args.skip_warmup:
        started = perf_counter()
        print(f"起動準備: {startup_model_id.model_name} のモデルと検索インデックスを読み込んでいます。", flush=True)
        in_usecase.warmup_search_cache(startup_model_id)
        in_usecase.get_catalog_count(startup_model_id)
        # Prepare category totals once before accepting users, so filtering does
        # not trigger a full-catalog aggregation during the first interaction.
        print("起動準備: 画像区分ごとの件数を集計しています。", flush=True)
        in_usecase.get_catalog_count(startup_model_id, ["general"])
        print(f"起動準備完了（{perf_counter() - started:.1f}秒）。ブラウザから接続できます。", flush=True)
    start_app(in_usecase, host=args.host, port=args.port)


def select_startup_model(models):
    if not models:
        raise ValueError("検索モデルが見つかりません。--meta-dir の場所とインデックスを確認してください。")
    return next(
        (item.id for item in models if item.id.model_name == "ViT-L-14" and item.id.pretrained == "openai"),
        models[0].id,
    )


if __name__ == "__main__":
    main()
