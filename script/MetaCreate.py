# coding: utf-8

import os
import shutil

import numpy as np
import torch
import tqdm
from PIL import Image

from script import util


def metaCreate(itemlist, args):

    print("metaCreate")
    model_dir_name = F'{itemlist.model[0]}-{itemlist.model["model"][1]}'
    meta_dir = F'{itemlist.metadataDir}/{model_dir_name}'
    image_dir = itemlist.metadataDir
    meta_files = itemlist.metadataFiles

    # モデルの読み込み
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = util.loadModel(itemlist.model, device=device)

    #　画像のパスを取得
    files = util.getImageFiles(image_dir)

    #　metaディレクトリをクリア
    os.makedirs(f'{meta_dir}', exist_ok=True)
    shutil.rmtree(f'{meta_dir}')
    os.makedirs(f'{meta_dir}', exist_ok=True)

    with torch.no_grad():
        pbar = tqdm.tqdm(files)
        for file in pbar:
            if not os.path.exists(f'{meta_dir}/{os.path.split(file)[1]}'):
                #　画像をオープン
                try:
                    data = Image.open(f'{image_dir}/{file}')
                except:
                    print(f'このファイルは開けません:{image_dir}/{file}')
                    continue
                image_features = util.encode_image(model, preprocess, data, device)
                #　ベクトルをmetaディレクトリに保存
                try:
                    os.makedirs(f'{meta_dir}/{os.path.split(file)[0]}')
                except:
                    pass
                try:
                    np.save(f'{meta_dir}/{file}.npy', image_features)
                except:
                    print(f'保存できません:{meta_dir}/{file}.npy')
    util.createPickleFile(files, meta_dir)
    metas = util.loadPickleFile(meta_dir)
    util.createIndexFile(metas, meta_dir)
    return ((file, 0) for file in files)

