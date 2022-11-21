import tqdm
from script import util
import re


def nameSearch(itemlist, args):

    print("nameSearch")  
    #　画像のリストを取得
    files = itemlist.imagesFiles

    # 類似度を計算する
    for file in tqdm.tqdm(files):
        # print(args.get("trueRegexp"))
        if args.get("trueRegexp")=="true":
            hasMatch = re.search(args.get("text"), file)
        else:
            hasMatch = args.get("text") in file

        if hasMatch:
            itemlist.setScore(file, 1.0)
        else:
            itemlist.setScore(file, 0.0)
    itemlist.sortScore()
    return 