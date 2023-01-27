import io
from flask import Flask, request, make_response
import json

from app.application.usecase import Usecase
from app.domain.domain_object import Item




app = Flask(__name__, static_folder='')

#エントリーポイント
def start_app(in_usecase : Usecase) -> Flask:

    #ユースケースのDI
    global usecase
    usecase = in_usecase

    #画像の読み込み
    usecase.load_items()
    
    #Flask実行
    print("Flask実行")
    app.run(debug=False)




# メインページを返す
@app.route('/')
def index():

    #ビューの指定をしている。
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()
    
    return text

#画像IDから画像を返す
@app.route("/image/<id>")
def load_image(id):

    #画像を取得して、レスポンスに詰め替えて返す。
    img = usecase.get_image(id)
    response = make_response(img.get_binary())
    response.headers.set('Content-Type', img.get_mime_type())
    return response

#画像IDから画像名を返す
@app.route("/image/<id>/name")
def get_path(id):
    
    if usecase.get_item_name(id):
        raise 
    return usecase.get_item_name(id)



#画像IDをすべて返す
@app.route("/image/all")
def get_all_image_id():
    dict = {"image":[{"id": id} for id, in usecase.get_ids()]}
    response = json.dumps(dict)
    return response

#文字列から検索して、画像IDとスコアを返す
@app.route("/search/text")
def search_text():
    
    text = request.args.get("text")
    print(text)
    result = usecase.search_from_text(text)
    dict = {"image":[{"id": id, "score": score} for id, score in result]}
    response = json.dumps(dict)
    return response

#画像から検索して、画像IDとスコアを返す
@app.route("/search/image?payload=<json>")
def search_image():
    # TODO
    return

#アップロードされた画像から検索して、画像IDとスコアを返す                                                                                     
@app.route("/search/uploadimage?payload=<json>")
def upload_image():
    # TODO
    return

