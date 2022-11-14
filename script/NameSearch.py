import tqdm
from script import util
import re


def nameSearch(setting, args):

    print("nameSearch")
    model_dir_name = F'{setting["model"][0]}-{setting["model"][1]}'
    meta_dir = F'{setting["meta_dir"]}/{model_dir_name}'
    image_dir = setting["image_dir"]
    
    #　画像のリストを取得
    files = util.getImageFiles(image_dir)

    # 類似度を計算する
    scores = []
    for file in tqdm.tqdm(files):
        # print(args.get("trueRegexp"))
        if args.get("trueRegexp")=="true":
            hasMatch = re.search(args.get("text"), file)
        else:
            hasMatch = args.get("text") in file
        
        if hasMatch:
            scores.append([file, 1.0])
        else:
            scores.append([file, 0.0])
    return sorted(scores, reverse=True, key=lambda x: x[1])