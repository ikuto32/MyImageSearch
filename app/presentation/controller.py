
import ast
from flask import Flask, request, make_response

import pathlib
import json
import base64
import mimetypes

from app.application.usecase import Usecase
from app.domain.domain_object import ImageItem, ImageId, ModelId, ModelItem, UploadImage, ResultImageItem, UploadText


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


@app.route("/<path:terget>")
def resource(terget: str):
    """ファイルを返す"""

    # ビューの指定をしている。
    pwd = pathlib.Path(__file__).parent
    with open(f'{pwd}/view/{terget}', encoding="UTF-8") as f:
        text = f.read()

    response = make_response(text)
    response.headers.set('Content-Type', mimetypes.guess_type(terget)[0])
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
    model_name: str = args.get("model_name")
    pretrained: str = args.get("pretrained")
    text: str = args.get("text")
    print({"parameter": {"model_name": model_name,
          "pretrained": pretrained, "text": text}})
    return from_result_to_json(usecase.search_text(ModelId(model_name, pretrained), UploadText(text)))


@app.route("/search/image", methods=["POST"])
def search_image():
    """画像から検索して、結果を返す"""

    # jsonを解釈
    json_obj = json.loads(request.data).get("params")

    # 検索モデルのIDを取得する。
    model_id: ModelId = ModelId(json_obj["model_name"], json_obj["pretrained"])

    # IDのリストを取得する。
    id_text_list: list[str] = json_obj["id"]
    id_list: list[ImageId] = list(map(ImageId, id_text_list))

    return from_result_to_json(usecase.search_image(model_id, id_list))


@app.route("/search/name", methods=["POST"])
def search_name():
    """名前から検索して、結果を返す"""
    print(request.data.decode('utf-8'))
    args = ast.literal_eval(request.data.decode('utf-8')).get("params")
    text: str = args.get("text")
    is_regexp: bool = (args.get("is_regexp") == "true")
    print({"parameter": {"text": text}})
    return from_result_to_json(usecase.search_name(UploadText(text), is_regexp))


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

# ============================================================


def from_image_item_to_json(value: ImageItem) -> str:

    obj = {"id": value.id.id, "name": value.display_name.name}
    return json.dumps(obj)


def from_image_item_list_to_json(value: list[ImageItem]) -> str:

    objs = map(lambda r: {"id": r.id.id, "name": r.display_name.name}, value)
    objs = list(objs)
    return json.dumps(objs)


def from_result_to_json(value: list[ResultImageItem]) -> str:

    objs = map(lambda r: {
        "item": {
            "id": r.item.id.id,
            "name": r.item.display_name.name
        },
        "score": r.score.score
    },
        value
    )

    objs = list(objs)
    return json.dumps(objs)


def from_model_id_to_json(value: list[ModelItem]) -> str:

    objs = map(lambda r: {"model_name": r.id.model_name,
               "pretrained": r.id.pretrained}, value)
    objs = list(objs)
    return json.dumps(objs)


# ============================================================

@app.route("/model_item")
def get_all_model_item():
    return from_model_id_to_json(usecase.get_all_model())
