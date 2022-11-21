import os
from flask import Flask, request, jsonify
import base64

import ast

from script import util, script

app = Flask(__name__, static_folder='')

class ItemList:
    def __init__(self, image_dir:str, metadata_dir:str, model) -> None:
        self.state = ""
        self.model = model
        self.imagesDir = image_dir
        self.metadataDir = metadata_dir
        self.imagesFiles = util.getImageFiles(image_dir=image_dir)
        model_dir_name = F'{self.model[0]}-{self.model[1]}'
        util.createPickleFile(image_filse=self.imagesFiles, meta_dir=F'{self.metadataDir}/{model_dir_name}')
        self.metadataFiles = util.getMetafilse(meta_dir=F'{self.metadataDir}/{model_dir_name}')
        self.items = {file : 0.0 for file in self.imagesFiles}
    
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


itemlist = ItemList(image_dir="./images", metadata_dir="./meta", model=('ViT-B-32', 'laion2b_s34b_b79k'))

@app.route('/')
def index():
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()

    return text

@app.route("/score")
def getScore():
    args = request.args
    return itemlist.getScore(item_name=args.get("metaName"))

@app.route("/images")
def getItems():
    args = request.args
    img_dict = {}
    items = itemlist.getItemsFromOrder(int(args.get("min")), int(args.get("max")))
    images_dir = itemlist.imagesDir
    for name, score in items:
        with open(F"{images_dir}/{name}", "rb") as image_file:
            img = image_file.read()
        file_type = os.path.splitext(name)[1][1:]
        base64_img = base64.b64encode(img).decode('utf-8')
        img_dict[name] = {"base64_img": base64_img, "score": score, "file_type": file_type}
    return jsonify(img_dict)

@app.route("/search", methods=["POST"])
def search():
    print(request.data.decode('utf-8'))
    args = ast.literal_eval(request.data.decode('utf-8'))
    print(args)
    out = script.onInputEnd(itemlist, args)
    return out

if __name__ == "__main__":
    app.run(debug=True)

