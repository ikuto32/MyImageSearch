# Qwen3-VL-EmbeddingとvLLM API

`create_index.py --search_backend qwen_vl` は、画像の埋め込みを別プロセスのOpenAI互換APIへ要求します。クライアントは画像の読み込み・SQLite保存・FAISS作成を担当し、Qwenの推論はvLLM側で行います。

## サーバーの準備

MyImageSearchとは別の、vLLMを実行できるLinux/WSL環境を用意してください。インストール条件とモデル対応は [vLLMの公式手順](https://docs.vllm.ai/en/stable/getting_started/installation/) と [マルチモーダル埋め込みの例](https://docs.vllm.ai/en/stable/examples/pooling/embed/) を参照してください。

起動例（bash）：

```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B \
  --served-model-name Qwen/Qwen3-VL-Embedding-2B \
  --runner pooling --host 127.0.0.1 --port 8000 \
  --dtype float16 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":4096,"max_pixels":262144}'
```

モデルやGPUに応じてメモリ・入力解像度の設定を調整します。`min_pixels` / `max_pixels` はvLLM起動時の設定です。クライアントの互換用引数 `--qwen-max-pixels` では変更できません。

## 接続とインデックス作成

まず画像1枚で接続を確認します。FAISSの学習を伴わないため、少数画像でのAPI確認に使えます。

```powershell
uv run python qwen3_vl_embed_image_fixed.py images/example.jpg --api-base http://127.0.0.1:8000/v1 --output .cache/embedding.npy
```

続いて画像ディレクトリからインデックスを作成します。次は少数画像（32枚以上）の動作確認用設定です。

```powershell
uv run python create_index.py --search_backend qwen_vl --search_model_id Qwen/Qwen3-VL-Embedding-2B --search_model_out_dim 2048 --qwen-api-base http://127.0.0.1:8000/v1 --batch_size 8 --qwen-api-concurrency 4 --num_workers 0 --disable-clip-metadata --use-existing-tags --nlist 1 --bits_per_code 4 --image_dir ./images --meta_dir ./clip_meta
```

`VLLM_API_BASE` / `VLLM_API_KEY` 環境変数でも接続先とキーを設定できます。キーをコードへ書き込まないでください。`/v1` の省略時はクライアントが補完します。WSLとホスト間でlocalhost転送を使えない場合は、その環境の接続可能なアドレスを指定します。

このAPI設定が適用されるのはインデックス作成と単画像CLIです。現在のWeb検索側はHugging Faceのモデルをローカルに読み込み、vLLM APIを使用しません。Qwenの重みと推論用メモリがWebアプリ側にも必要です。APIで作成したインデックスとの組み合わせでは、モデル・次元・前処理条件と検索結果の整合を個別に検証してください。モデル固有の仕様は [Qwenのモデルカード](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) を参照してください。
