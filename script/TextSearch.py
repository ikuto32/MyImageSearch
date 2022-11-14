# coding: utf-8

import os

import numpy as np

from script import util


def textSearch(setting, args, metas):

    print("textSearch")
    model_dir_name = F'{setting["model"][0]}-{setting["model"][1]}'
    meta_dir = F'{setting["meta_dir"]}/{model_dir_name}'
    image_dir = setting["image_dir"]
    
    # モデルの読み込み
    model, _, _ = util.loadModel(setting["model"], device="cpu")

    # テキストの埋め込みを計算
    features = util.encode_text(model, args.get("text"))

    # indexを読み込み
    index = util.loadIndexFile(meta_dir)

    # 類似度を計算する
    scores = util.eval(metas, index, features)
    return sorted(scores, reverse=True, key=lambda x: x[1])
