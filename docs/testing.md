# 検証ガイド

リポジトリのルートで実行します。個人の保存先や計測ログは公開せず、出力は `.cache/` に保存してください。

## 自動テスト

```powershell
uv run python tests/run_tests.py
node --test tests/test_ui_state.cjs tests/test_catalog_scroll.cjs tests/test_i18n.cjs
```

Pythonテストは各ファイルを別プロセスで実行し、MLライブラリのスタブが別のテストへ混入するのを防ぎます。ブラウザ用の画像は `tests/fixtures/image.png`（メタデータを持たない2色のテスト画像）を使用し、利用者の画像データを必要としません。

## ブラウザでの操作・レイアウト検証

Node.jsとPlaywrightを用意します。

```powershell
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node scripts/review_responsive.cjs
node scripts/review_dialogs.cjs
node scripts/review_scroll.cjs
node scripts/review_scroll.cjs --position
```

これらは固定のAPI応答を返すローカルサーバーを起動します。レスポンシブ検証では16幅・日英・一覧と検索・選択件数を切り替えます。保存確認、画像詳細の遅延と再試行、スクロール、現在位置の表示は個別スクリプトで確認します。合成データの結果は実データの検索性能を示しません。

既存のChromeを使う場合は `CHROME_PATH` に実行ファイル、別の場所にあるPlaywrightを使う場合は `PLAYWRIGHT_MODULE` にパッケージの場所を設定できます。画面入力の自動テストに加え、iPadなどの実機でも、指の追加・離脱・斜め移動・慣性スクロールを確認してください。

## 用意した軽量データを検証する

[README](../README.md) の手順で `images/` と `clip_meta/` を用意してから実行します。

```powershell
uv run python scripts/validate_local.py
uv run python scripts/validate_local.py --with-text
```

既定のViT-L-14/openaiを対象に、一覧・画像配信・検索API・保存クエリの再実行を読み取り専用で検証します。`--with-text` は実際のCLIP推論を含み、未取得のモデルはダウンロードされます。`--all-models` を追加すると、他の登録モデルのインデックスと保存クエリも確認します。

## 大規模データの性能とUX

画像とインデックスは同じデータセットに対応するものを指定します。準備が完了するまで待ち、モデル・件数・使用メモリ・実行環境・キャッシュの有無を記録して比較してください。

```powershell
uv run python scripts/review_large_data.py --image-dir "D:/photos" --meta-dir "D:/image-index" --serve --port 5001
```

起動準備、一覧、検索、画像配信を測定し、`.cache/large-review.json` に出力します。`--serve` は準備したサーバーをブラウザレビュー用に維持します。別のターミナルで実行します。

```powershell
$env:UX_BASE_URL = "http://127.0.0.1:5001"
node scripts/review_ui.cjs
node scripts/review_scroll.cjs --url http://127.0.0.1:5001 --position
```

先頭・中央・末尾への移動、追加取得中の位置保持、検索中の待ち時間、小さい画面での操作を確認します。初回の区分集計や離れた位置への移動は、近傍ページの取得より時間がかかることがあります。起動時間はモデル準備だけでなく、プロセス開始から接続可能になるまでを測ってください。

部分的な計測には次のスクリプトを使えます。

```powershell
uv run python scripts/benchmark_startup.py --meta-dir "D:/image-index"
uv run python scripts/benchmark_catalog_filters.py --meta-dir "D:/image-index"
uv run python scripts/benchmark_metadata.py
```

`benchmark_startup.py` は実DBの一覧初期化、`benchmark_catalog_filters.py` は件数・区分・ページ取得を測定します。`benchmark_metadata.py` は合成100万件のメタデータ参照のみを対象とし、画像検索全体の測定ではありません。インデックス更新中の同時実行は避け、更新後はサーバーを再起動してください。
