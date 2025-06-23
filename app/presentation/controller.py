
import ast
import logging
from flask import Flask, request, make_response, send_file

import pathlib
import json
import base64
import mimetypes

from app.application.usecase import Usecase
from app.domain.domain_object import ImageItem, ImageId, ModelId, ModelItem, ResultImageItemList, UploadImage, ResultImageItem, UploadText


# ============================================================
# 初期化処理

app = Flask(__name__, static_folder=None)
usecase: Usecase = None


def start_app(in_usecase: Usecase):
    """エントリーポイント"""

    # ユースケースのDI
    global usecase
    usecase = in_usecase

    # text/javascript が text/plain になる問題の対策
    # https://bugs.python.org/issue43975
    mimetypes.add_type("text/javascript", ".js", True)

    # ログレベルをWARNING以上に設定することで、INFOレベルのログを非表示にします
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    # Flask実行
    print(app.url_map)
    app.run(debug=False, port=80, host="0.0.0.0")


# ============================================================
# エンドポイント

@app.route('/')
def index():
    """メインページを返す"""

    # ビューの指定をしている。
    pwd = pathlib.Path(__file__).parent
    with open(f'{pwd}/view/index.html', encoding="UTF-8") as f:
        text = f.read()

    return text


@app.route("/<path:target>")
def resource(target: str):
    """ファイルを返す"""

    # ビューの指定をしている。
    pwd = pathlib.Path(__file__).parent
    base_path = pwd / 'view'
    full_path = pathlib.Path(base_path / target).resolve()
    if not str(full_path).startswith(str(base_path.resolve())):
        raise Exception("Access to the specified file is not allowed.")
    with open(full_path, encoding="UTF-8") as f:
        text = f.read()

    response = make_response(text)
    response.headers.set('Content-Type', mimetypes.guess_type(target)[0])
    return response


# ------------------------------------------------------------


@app.route("/image_item")
def get_all_image_item():
    """画像項目をすべて返す"""

    return from_image_item_list_to_json(usecase.get_all_image_item()[0:10000])


@app.route("/image_item/<id>")
def get_image_item(id: str):
    """画像IDから画像項目を返す"""

    return from_image_item_to_json(usecase.get_image_item(ImageId(id)))


@app.route("/image_item/<id>/image")
def get_image(id: str):
    """画像IDから画像を返す"""

    # 画像を取得して、レスポンスに詰め替えて返す。
    img = usecase.get_image(ImageId(id))
    response = make_response(img.binary)
    response.headers.set('Content-Type', img.content_type)
    return response


# ------------------------------------------------------------


@app.route("/search/text", methods=["POST"])
def search_text():
    """文字列から検索して、結果を返す"""
    args = ast.literal_eval(request.data.decode('utf-8')).get("params")
    model_id: ModelId = ModelId(args.get("model_name"), args.get("pretrained"))
    text: str = args.get("text")
    # 美感スコアの重要度を取得する。
    aesthetic_quality_beta: float = args.get("aesthetic_quality_beta")
    aesthetic_quality_range = args.get("aesthetic_quality_range")
    aesthetic_model_name = args.get("aesthetic_model_name")
    print({"parameter": {"model_id": model_id, "text": text, "aesthetic_quality_beta": aesthetic_quality_beta, "aesthetic_quality_range": aesthetic_quality_range, "aesthetic_model_name": aesthetic_model_name}})
    result = usecase.search_text(model_id, UploadText(text), aesthetic_quality_beta, aesthetic_quality_range[0], aesthetic_quality_range[1], aesthetic_model_name)
    return from_result_to_json(result)


@app.route("/search/image", methods=["POST"])
def search_image():
    """画像から検索して、結果を返す"""

    # jsonを解釈
    json_obj = json.loads(request.data).get("params")

    # 検索モデルのIDを取得する。
    model_id: ModelId = ModelId(json_obj["model_name"], json_obj["pretrained"])

    # 美感スコアの重要度を取得する。
    aesthetic_quality_beta: float = json_obj["aesthetic_quality_beta"]

    aesthetic_quality_range = json_obj["aesthetic_quality_range"]

    aesthetic_model_name = json_obj["aesthetic_model_name"]

    # IDのリストを取得する。
    id_text_list: list[str] = json_obj["id"]
    id_list: list[ImageId] = list(map(ImageId, id_text_list))
    result = usecase.search_image(model_id, id_list, aesthetic_quality_beta, aesthetic_quality_range[0], aesthetic_quality_range[1], aesthetic_model_name)
    return from_result_to_json(result)


@app.route("/search/name", methods=["POST"])
def search_name():
    """名前から検索して、結果を返す"""
    print(request.data.decode('utf-8'))
    args = ast.literal_eval(request.data.decode('utf-8')).get("params")
    # 検索モデルのIDを取得する。
    model_id: ModelId = ModelId(args.get("model_name"), args.get("pretrained"))
    text: str = args.get("text")
    is_regexp: bool = (args.get("is_regexp") == "true")
    # 美感スコアの重要度を取得する。
    aesthetic_quality_beta: float = args.get("aesthetic_quality_beta")
    aesthetic_quality_range = args.get("aesthetic_quality_range")
    aesthetic_model_name = args.get("aesthetic_model_name")
    print({"parameter": {"text": text}})
    result = usecase.search_name(model_id, UploadText(text), is_regexp, aesthetic_quality_beta, aesthetic_quality_range[0], aesthetic_quality_range[1], aesthetic_model_name)
    return from_result_to_json(result)


@app.route("/search/uploadimage", methods=["POST"])
def search_upload_image():
    """アップロードされた画像から検索して、結果を返す。"""

    payload: str = request.args.get("payload")

    # jsonを解釈し、値を取り出す。
    json_obj = json.loads(payload)
    model_id: ModelId = ModelId(json_obj["model"], json_obj["pretrained"])
    base64_text: str = json_obj["base64"]
    content_type: str = json_obj(["content_type"])

    # base64を画像に変換
    binary = base64.b64decode(base64_text)
    image: UploadImage = UploadImage(binary, content_type)

    return from_result_to_json(usecase.search_upload_image(model_id, image))


@app.route("/search/random", methods=["POST"])
def search_random():
    """乱数から検索して、結果を返す"""

    # jsonを解釈
    json_obj = json.loads(request.data).get("params")

    # 検索モデルのIDを取得する。
    model_id: ModelId = ModelId(json_obj["model_name"], json_obj["pretrained"])

    # 美感スコアの重要度を取得する。
    aesthetic_quality_beta: float = json_obj["aesthetic_quality_beta"]

    aesthetic_quality_range = json_obj["aesthetic_quality_range"]

    aesthetic_model_name = json_obj["aesthetic_model_name"]

    # IDのリストを取得する。
    result = usecase.search_random(model_id, aesthetic_quality_beta, aesthetic_quality_range[0], aesthetic_quality_range[1], aesthetic_model_name)
    return from_result_to_json(result)


@app.route("/search/query", methods=["POST"])
def search_query():
    """クエリから検索して、結果を返す"""

    # jsonを解釈
    json_obj = json.loads(request.data).get("params")

    # 検索モデルのIDを取得する。
    model_id: ModelId = ModelId(json_obj["model_name"], json_obj["pretrained"])

    search_query = json_obj["search_query"]

    # 美感スコアの重要度を取得する。
    aesthetic_quality_beta: float = json_obj["aesthetic_quality_beta"]

    aesthetic_quality_range = json_obj["aesthetic_quality_range"]

    aesthetic_model_name = json_obj["aesthetic_model_name"]

    # IDのリストを取得する。
    result = usecase.search_query(model_id, search_query, aesthetic_quality_beta, aesthetic_quality_range[0], aesthetic_quality_range[1], aesthetic_model_name)
    return from_result_to_json(result)


@app.route("/search/queryaddtext", methods=["POST"])
def add_text_features():
    """クエリにテキストの特徴を足してから検索して、結果を返す"""

    # jsonを解釈
    json_obj = json.loads(request.data).get("params")

    # 検索モデルのIDを取得する。
    model_id: ModelId = ModelId(json_obj["model_name"], json_obj["pretrained"])

    text: str = json_obj["text"]

    search_query = json_obj["search_query"]

    strength = json_obj["features_strength"]

    # 美感スコアの重要度を取得する。
    aesthetic_quality_beta: float = json_obj["aesthetic_quality_beta"]

    aesthetic_quality_range = json_obj["aesthetic_quality_range"]

    aesthetic_model_name = json_obj["aesthetic_model_name"]

    # IDのリストを取得する。
    result = usecase.add_text_features(model_id, UploadText(text), search_query, strength, aesthetic_quality_beta, aesthetic_quality_range[0], aesthetic_quality_range[1], aesthetic_model_name)
    return from_result_to_json(result)


@app.route("/search/tags", methods=["POST"])
def search_tags():
    """タグから検索して、結果を返す"""

    # jsonを解釈
    json_obj = json.loads(request.data).get("params")

    # 検索モデルのIDを取得する。
    model_id: ModelId = ModelId(json_obj["model_name"], json_obj["pretrained"])

    text: str = json_obj.get("text")
    is_regexp: bool = (json_obj.get("is_regexp") == "true")

    # 美感スコアの重要度を取得する。
    aesthetic_quality_beta: float = json_obj["aesthetic_quality_beta"]

    aesthetic_quality_range = json_obj["aesthetic_quality_range"]

    aesthetic_model_name = json_obj["aesthetic_model_name"]

    # IDのリストを取得する。
    result = usecase.search_tags(model_id, UploadText(text), is_regexp, aesthetic_quality_beta, aesthetic_quality_range[0], aesthetic_quality_range[1], aesthetic_model_name)
    return from_result_to_json(result)

@app.route("/download_images_zip", methods=["POST"])
def download_images_zip():
    json_obj = request.get_json().get("params")
    id_list = json_obj.get("ids", [])

    zip_buffer = usecase.get_images_zip(id_list)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='images.zip'
    )
# ============================================================


def from_image_item_to_json(value: ImageItem) -> str:

    obj = {"id": value.id.id, "name": value.display_name.name, "tags": value.tags.tags}
    return json.dumps(obj)


def from_image_item_list_to_json(value: list[ImageItem]) -> str:

    objs = map(lambda r: {"id": r.id.id, "name": r.display_name.name, "tags": r.tags.tags}, value)
    objs = list(objs)
    return json.dumps(objs)


def from_result_to_json(value: ResultImageItemList) -> str:

    objs = map(lambda r: {
        "item": {
            "id": str(r.item.id.id),
            "name": str(r.item.display_name.name),
            "tags": str(r.item.tags.tags)
        },
        "score": float(r.score.score)
    },
        value.list
    )

    return json.dumps({
        "list": list(objs),
        "search_query": value.search_query
    })


def from_result_list_to_json(value: list[ResultImageItem]) -> str:

    objs = map(lambda r: {
        "item": {
            "id": str(r.item.id.id),
            "name": str(r.item.display_name.name),
            "tags": str(r.item.tags.tags)
        },
        "score": float(r.score.score)
    },
        value
    )

    return json.dumps(list(objs))


def from_model_id_to_json(value: list[ModelItem]) -> str:

    objs = map(lambda r: {"model_name": r.id.model_name,
               "pretrained": r.id.pretrained}, value)
    objs = list(objs)
    return json.dumps(objs)


# ============================================================

@app.route("/model_item")
def get_all_model_item():
    return from_model_id_to_json(usecase.get_all_model())
