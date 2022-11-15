from flask import Flask, request
from script import util

app = Flask(__name__, static_folder='')

class ItemList:
    def __init__(self, image_dir:str) -> None:
        self.imagesFiles = util.getImageFiles(image_dir=image_dir)
        self.item = {file : 0.0 for file in self.imagesFiles}
    
    def setScore(self, item_name:str, score:float):
        
        self.item[item_name] = score
    
    def getScore(self, item_name:str):
        return self.item.get(item_name)
    
    def getimagesFiles(self):
        return self.imagesFiles


itemlist = ItemList(image_dir="images")

@app.route('/')
def index():
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()

    return text

@app.route("/score")
def getScore():
    args = request.args
    return itemlist.getScore(args.get("metaName"))

if __name__ == "__main__":
    app.run(debug=True)

