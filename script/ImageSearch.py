# coding: utf-8

import numpy as np

from script import util


def imageSearch(setting, args, metas):

    print("ImageSearch")
    model_dir_name = F'{setting["model"][0]}-{setting["model"][1]}'
    meta_dir = F'{setting["meta_dir"]}/{model_dir_name}'
    image_dir = setting["image_dir"]

    select_image_paths = args.get("metaNames").split(",")

    if not select_image_paths:
        return

    #　選択した画像のmetaを結合する
    temp = []
    for select_image_path in select_image_paths:
        temp.append(util.loadMeta(meta_dir, select_image_path))
        features = np.vstack(temp)


    # indexを読み込み
    index = util.loadIndexFile(meta_dir)

    # 類似度を計算する
    scores = util.eval(metas, index, features)
    return sorted(scores, reverse=True, key=lambda x: x[1])

