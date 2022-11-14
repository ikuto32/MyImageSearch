# coding: utf-8
import numpy as np

from script import util


def textAndImageSearch(setting, args, metas):
    print("TextAndImageSearch")
    model_dir_name = F'{setting["model"][0]}-{setting["model"][1]}'
    meta_dir = F'{setting["meta_dir"]}/{model_dir_name}'
    image_dir = setting["image_dir"]

    
    select_image_paths = args.get("metaNames").split(",")

    if not select_image_paths:
        return
    
    #　選択した画像のmetaの平均を求める
    temp = []
    for select_image_path in select_image_paths:
        temp.append(util.loadMeta(meta_dir, select_image_path))
        image = np.mean(np.vstack(temp), axis=0, keepdims=True)
    # モデルの読み込み
    model, _, _ = util.loadModel(setting["model"], device="cpu")
    # テキストの埋め込みを計算
    text = util.encode_text(model, args.get("text"))

    p = float(args.get("ratio"))
    features = text*p + image*(1-p)

    # indexを読み込み
    index = util.loadIndexFile(meta_dir)

    # 類似度を計算する
    scores = util.eval(metas, index, features)
    return sorted(scores, reverse=True, key=lambda x: x[1])

