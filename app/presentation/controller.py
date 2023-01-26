import io
from flask import Flask, request, make_response

import json
import base64

from app.application.usecase import Usecase
from app.domain.domain_object import Image, Item, ItemId, SearchResult, SearchText


#============================================================


app = Flask(__name__, static_folder='')
usecase : Usecase = None

def startApp(in_usecase : Usecase):
    """エントリーポイント"""

    #ユースケースのDI
    global usecase
    usecase = in_usecase

    #画像の読み込み
    usecase.load_items()
    
    #Flask実行
    app.run(debug=True)



#============================================================


@app.route('/')
def index():
    """メインページを返す"""

    #ビューの指定をしている。
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()
    
    return text

@app.route("/image/<id>")
def load_image(id : str):
    """画像IDから画像を返す"""

    #画像を取得して、レスポンスに詰め替えて返す。
    img = usecase.get_image(ItemId(id))
    response = make_response(img.get_binary())
    response.headers.set('Content-Type', img.get_mime_type())
    return response


@app.route("/image/<id>/name")
def get_item_name(id : str):
    """画像IDから画像名を返す"""

    #TODO 例外処理をする
    if usecase.get_item_name(ItemId(id)):
        raise 
    return usecase.get_item_name(id)



@app.route("/search/all")
def search_all():
    """画像IDをすべて返す"""

    text = to_json(usecase.search_all())
    return text

@app.route("/search/text")
def search_text(text : str):
    """文字列から検索して、画像IDを返す"""
    
    text = to_json(usecase.search_from_text(SearchText(text)))
    return text

@app.route("/search/image")
def search_image(payload : str):
    """画像から検索して、画像IDを返す"""

    #jsonを解釈し、IDのリストを取得する。
    json_obj = json.loads(payload)
    id_text_list : list[str] = json_obj["id"]
    id_list : list[ItemId] = map(ItemId, id_text_list)
    
    text = to_json(usecase.search_from_image(id_list))
    return text
                                       
@app.route("/search/uploadimage")
def search_upload_image(payload : str):
    """アップロードされた画像から検索して、画像IDを返す。"""

    #jsonを解釈し、値を取り出す。
    json_obj = json.loads(payload)
    base64_text : str = json_obj["base64"]
    mime_type : str = json_obj(["mime_type"])

    #base64を画像に変換
    binary = base64.b64decode(base64_text)
    image : Image = Image(binary, mime_type)

    text = to_json(usecase.search_from_upload_image(image))
    return text



#============================================================



def to_json(result : SearchResult):

    objs = map(lambda i: {"id": i.id.id, "score": i.score.score}, result)
    return json.dumps(objs)



#============================================================

