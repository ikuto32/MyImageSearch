# coding: utf-8

import os

import numpy as np

from script import util


def textSearch(itemlist, args):

    print("textSearch")
    model_dir_name = F'{itemlist.model[0]}-{itemlist.model[1]}'
    meta_dir = F'{itemlist.metadataDir}/{model_dir_name}'
    image_dir = itemlist.metadataDir
    meta_files = itemlist.metadataFiles
    
    # モデルの読み込み
    model, _, _ = util.loadModel(itemlist.model, device="cpu")

    # テキストの埋め込みを計算
    features = util.encode_text(model, args.get("text"))

    # indexを読み込み
    index = util.loadIndexFile(meta_dir)

    # 類似度を計算する
    scores = util.eval(meta_files, index, features)
    for name, score in scores:
        itemlist.setScore(name, score)
    itemlist.sortScore()

    return ""
