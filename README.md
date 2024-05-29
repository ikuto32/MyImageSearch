# MyImageSearch

MyImageSearch は、OpenAI の [CLIP](https://openai.com/index/clip/) (Contrastive Language-Image Pre-Training) モデルに基づいて、テキストクエリを使用してローカルマシン上の画像を検索し、画像を見つけることができるWebアプリケーションです。

## Backend
画像を[openCLIP](https://github.com/mlfoundations/open_clip)を用いてembeddingsに変換し、SQliteデータベースで管理している。検索には[faiss](https://github.com/facebookresearch/faiss)を使用して"index"を作成し、高速なベクトル検索を行っている。ウェブアプリケーションの実装には [Flask](https://flask.palletsprojects.com/en/3.0.x/)フレームワークを使用した。

## Frontend
アプリケーションのフロントエンドは、人気のマテリアルデザインコンポーネントフレームワークである[Vue.js](https://vuejs.org/)と[Vuetify](https://vuetifyjs.com/en/)を使用して開発した。

## Technologies
このプロジェクトで使用される主な言語は、Python、JavaScript、HTML です。

# Setup
次の手順に従って、winマシンで MyImageSearch をセットアップして実行します。
## Create a new Conda environment:
```
conda create -n MyImageSearch python=3.9
```
## Activate the environment:
```
conda activate MyImageSearch
```

## Install environments:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c pytorch faiss-cpu
conda install -c anaconda pandas
pip install Flask
pip install open_clip_torch
```
## Run the application:
```
python app.py
```
## Access UI:
[localhost](http://localhost)にブラウザからアクセスします。

# TODO
- 動作のステータス表示する
- CoCa model
