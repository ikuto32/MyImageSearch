import io
from flask import Flask, request, make_response
import json

from app.application.usecase import Usecase
from app.domain.domain_object import Item




app = Flask(__name__, static_folder='')

#エントリーポイント
def startApp(in_usecase : Usecase):

    #ユースケースのDI
    global usecase
    usecase : Usecase = in_usecase

    #画像の読み込み
    usecase.load_items()
    
    #Flask実行
    app.run(debug=True)




# メインページを返す
@app.route('/')
def index():

    #ビューの指定をしている。
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()
    
    return text

#画像IDから画像を返す
@app.route("/image/<id>")
def loadImage(id):

    #画像を取得して、レスポンスに詰め替えて返す。
    img = usecase.get_image(id)
    response = make_response(img.get_binary())
    response.headers.set('Content-Type', img.get_mime_type())
    return response

#画像IDから画像名を返す
@app.route("/image/<id>/name")
def getPath(id):
    
    #TODO 警告 例外処理が不足
    return usecase.get_image_name(id)



#画像IDをすべて返す
@app.route("/image/all")
def getAllImageId():
    dict = {"image":[{"id": id} for id, in itemlist.itemIDs.keys()]}
    response = json.dumps(dict)
    return response

#文字列から検索して、画像IDを返す
@app.route("/search/text")
def searchText():
    
    text = request.args.get("text")
    print(text)
    scores = TextSearch.textSearch(itemlist, text)
    dict = {"image":[{"id": itemlist.itemIDs.get(name), "score": score} for name, score in scores]}
    response = json.dumps(dict)
    return response

#画像から検索して、画像IDを返す
@app.route("/search/image?payload=<json>")
def searchImage():
    # TODO
    return

#アップロードされた画像から検索して、画像IDを返す。                                                                                     
@app.route("/search/uploadimage?payload=<json>")
def uploadImage():
    # TODO
    return

