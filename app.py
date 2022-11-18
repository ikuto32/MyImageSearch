import io
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from script import util

app = Flask(__name__, static_folder='')

class ItemList:
    def __init__(self, image_dir:str) -> None:
        self.imagesDir = image_dir
        self.imagesFiles = util.getImageFiles(image_dir=image_dir)
        self.items = {file : 0.0 for file in self.imagesFiles}
    
    def setScore(self, item_name:str, score:float):
        self.items[item_name] = score
    
    def getScore(self, item_name:str):
        return self.items.get(item_name)
    
    def getimagesFiles(self):
        return self.imagesFiles
    
    def sortScore(self):
        return sorted(self.items, key=lambda x:x[1])

    def getFromOrder(self, order:int):
        return list(self.items.keys())[order]

    def getItemsFromOrder(self, min:int, max:int):
        return dict(list(self.items.items())[min:max])


itemlist = ItemList(image_dir="images")

@app.route('/')
def index():
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()

    return text

@app.route("/score")
def getScore():
    args = request.args
    return itemlist.getScore(item_name=args.get("metaName"))

@app.route("/image")
def getImage():
    args = request.args
    response = itemlist.getFromOrder(order=int(args.get("order")))
    return response

@app.route("/images")
def getItems():
    args = request.args
    response = itemlist.getItemsFromOrder(int(args.get("min")), int(args.get("max")))
    return jsonify(response)
if __name__ == "__main__":
    app.run(debug=True)

