# coding: utf-8

import numpy as np

from script import util


def imageSearch(itemlist, args):

    print("ImageSearch")
    model_dir_name = F'{itemlist.model[0]}-{itemlist.model[1]}'
    meta_dir = F'{itemlist.metadataDir}/{model_dir_name}'
    image_dir = itemlist.metadataDir
    meta_files = itemlist.metadataFiles

    select_image_paths = args.get("meta_names").split(",")

    if not select_image_paths:
        return

    #　選択した画像のmetaを結合する
    temp = []
    for select_image_path in select_image_paths:
        try:
            meta = util.loadMeta(meta_dir, select_image_path)
            temp.append(meta)
        except:
            pass
        
    try:
        features = np.vstack(temp)
    except:
        return


    # indexを読み込み
    index = util.loadIndexFile(meta_dir)

    # 類似度を計算する
    scores = util.eval(meta_files, index, features)
    for name, score in scores:
        itemlist.setScore(name, score)
    itemlist.sortScore()
    return scores

