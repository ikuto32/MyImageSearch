import io
from flask import Flask, request, make_response
import json

from app.application.usecase import Usecase
from app.domain.domain_object import Item, ItemId, SearchResult


#============================================================


app = Flask(__name__, static_folder='')
usecase : Usecase = None

#エントリーポイント
def startApp(in_usecase : Usecase):

    #ユースケースのDI
    global usecase
    usecase = in_usecase

    #画像の読み込み
    usecase.load_items()
    
    #Flask実行
    app.run(debug=True)



#============================================================


# メインページを返す
@app.route('/')
def index():

    #ビューの指定をしている。
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()
    
    return text

#画像IDから画像を返す
@app.route("/image/<id>")
def loadImage(id : str):

    #画像を取得して、レスポンスに詰め替えて返す。
    img = usecase.get_image(ItemId(id))
    response = make_response(img.get_binary())
    response.headers.set('Content-Type', img.get_mime_type())
    return response

#画像IDから画像名を返す
@app.route("/image/<id>/name")
def getPath(id : str):
    
    if usecase.get_item_name(ItemId(id)):
        raise 
    return usecase.get_item_name(id)



#画像IDをすべて返す
@app.route("/search/all")
def getAllImageId():

    text = to_json(usecase.search_all())
    return text

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



#============================================================



def to_json(result : SearchResult):

    objs = map(lambda i: {"id": i.id.id, "score": i.score.score}, result)
    return json.dumps(objs)



#============================================================

