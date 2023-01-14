import io
from flask import Flask, request, make_response
import json

import script.application.item_service as item_service
from script.domain.domain_object import Config, Item




app = Flask(__name__, static_folder='')


#エントリーポイント
def startApp(config : Config):

    #項目の用意
    item_service.load(config)
    
    #Flask実行
    app.run(debug=True)




# メインページを返す
@app.route('/')
def index():
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()

    return text

# 画像IDから画像を返す
@app.route("/image/<id>")
def loadImage(id):

    
    item : Item = item_service.get_item_from_id(id)

    path = f'{itemlist.imagesDir}/{itemlist.itemIDs.get(id)}'
    img_bin = open(path, 'rb').read()
    response = make_response(img_bin)
    response.headers.set('Content-Type', request.content_type)
    return response

#画像IDから画像パスを返す
@app.route("/image/<id>/path")
def getPath(id):
    path = f'{itemlist.imagesDir}/{itemlist.itemIDs.get(id)}'
    response = json.dumps({"path":path})
    return response

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

