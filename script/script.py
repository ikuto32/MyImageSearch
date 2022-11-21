from app import ItemList
from script import CopyImages, ImageSearch, MetaCreate, TextAndImageSearch, TextSearch, NameSearch


#コールバック関数　引数は、発生したイベントの情報
def onInputEnd(itemlist:ItemList, args):

    print("event: " + str(args))
    print("params" + str(args.get("trigger")))

    #イベントの名前で分岐
    if args.get("trigger") == "TextSearch" :

        #文字列に対する関連度で検索をする
        return TextSearch.textSearch(itemlist, args)


    elif args.get("trigger") == "ImageSearch" :
        
        #画像に対する関連度で検索をする
        return ImageSearch.imageSearch(itemlist, args)


    elif args.get("trigger") == "NameSearch" :
        
        #文字列一致検索をする
        return NameSearch.nameSearch(itemlist, args)
        

    elif args.get("trigger") == "MetaCreate" :

        #metaを作る
        return MetaCreate.metaCreate(itemlist, args)

    elif args.get("trigger") == "TextAndImageSearch":

        #文字列と画像で検索をする
        return TextAndImageSearch.textAndImageSearch(itemlist, args)

    elif args.get("trigger") == "copyImages":

        #画像をコピーする
        return CopyImages.copyImages(itemlist, args) 

    elif args.get("trigger") == "":  
        pass

    print("script-error:対応していないイベント " + str(args.get("trigger")))
    return None


if __name__ == '__main__':
    onInputEnd()
