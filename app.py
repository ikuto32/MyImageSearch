import io
from flask import Flask, request, make_response
import json

from script import util, TextSearch

app = Flask(__name__, static_folder='')

class ItemList:
    def __init__(self, image_dir:str, metadata_dir:str, out_dir:str, model) -> None:
        self.state = ""
        self.model = model
        self.imagesDir = image_dir
        self.metadataDir = metadata_dir
        self.outDir = out_dir
        self.imagesFiles = util.getImageFiles(image_dir=image_dir)
        model_dir_name = F'{self.model[0]}-{self.model[1]}'
        util.createPickleFile(image_filse=self.imagesFiles, meta_dir=F'{self.metadataDir}/{model_dir_name}')
        self.metadataFiles = util.getMetafilse(meta_dir=F'{self.metadataDir}/{model_dir_name}')
        self.items = {file : 0.0 for file in self.imagesFiles}
        self.itemIDs = {str(id) : file for id, file in enumerate(self.imagesFiles)}
    
    def setScore(self, item_name:str, score:float):
        self.items[item_name] = score

    def setScores(self, item_names:str, scores:float):
        for item_name, score in zip(item_names, scores):
            self.setScore(item_name=item_name, score=score)
    
    def getScore(self, item_name:str):
        return self.items.get(item_name)
    
    def getimagesFiles(self):
        return self.imagesFiles
    
    def sortScore(self):
        print("sort")
        self.items = dict(sorted(self.items.items(), key=lambda x:x[1], reverse=True))
        return self.items

    def getFromOrder(self, order:int):
        return list(self.items.keys())[order]

    def getItemsFromOrder(self, min:int, max:int):
        return list(self.items.items())[min:max]
    
    def getMetadata(self, item_name:str):
        return util.loadMeta(item_name, self.imagesDir)


itemlist = ItemList(image_dir="./images", metadata_dir="./meta", out_dir="./out", model=('ViT-B-32', 'laion2b_s34b_b79k'))

# メインページを返す
@app.route('/')
def index():
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()

    return text

# 画像IDから画像を返す
@app.route("/image/<id>")
def loadImage(id):
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


if __name__ == "__main__":
    app.run(debug=True)

