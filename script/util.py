import os
import pathlib
import pickle

import numpy as np
import torch
import tqdm
import faiss
import open_clip

metasPickleFile = "metafiles.pickle"
metasFaissIndexFile = "metafiles.index"

# コサイン類似度
def cosine_similarity(v1, v2):
    v1, v2 = v1[0], v2[0]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# faissでmetaのindexファイルを作成する
def createIndexFile(metas, metadir):
    print("createIndexFile")
    metalist = list(metas.values())
    a = np.squeeze(np.asarray(metalist)).astype(np.float32)
    faiss.normalize_L2(a)
    dim = a.shape[1]
    q = faiss.IndexFlatIP(dim)
    nlist = 64
    M = 32
    bits_per_code = 4
    index = faiss.IndexIVFPQ(q, dim, nlist, M, bits_per_code, faiss.METRIC_INNER_PRODUCT)
    print("train")
    index.train(a)
    print("add")
    index.add(a)
    print("write")
    faiss.write_index(index, f'{metadir}/{metasFaissIndexFile}')

# faissでmetaのindexファイルを読み込む
def loadIndexFile(metadir):
    return faiss.read_index(f'{metadir}/{metasFaissIndexFile}')

# コサイン類似度を計算する
def eval(files, index, features, k=2048):
    features = features.astype(np.float32)
    faiss.normalize_L2(features)
    if k > len(files):
        k = len(files)
    
    index.nprobe = 64

    print(index.ntotal)
    D, I = index.search(features, k=k)
    
    temp = [0 for i in files]
    for i, d in zip(I, D):
        for item_id, distance in zip(i, d):
            temp[item_id] += distance
    scores = []
    for i, score in enumerate(temp):
        scores.append([files[i], score])
    return scores

#　metaのpickleファイルを作成する
def createPickleFile(files, metadir):
    print("Create pickle file")
    pbar = tqdm.tqdm(files)
    metas = {}
    for file in pbar:
        filename = f'{file}'
        try:
            metas[filename] = loadMeta(metadir, filename)
        except Exception:
            print(f"このmetaファイルは読み込めません:{filename}")
    with open(f'{metadir}/{metasPickleFile}', 'wb') as f:
        pickle.dump(metas, f)

#　metaのpickleファイルを読み込みする
def loadPickleFile(metadir):
    with open(f'{metadir}/{metasPickleFile}', 'rb') as f:
        return pickle.load(f)

# metaのpickleファイルが存在するか？
def PickleFileExists(meta_dir):
    return os.path.exists(f'{meta_dir}/{metasPickleFile}')

# metaを読み込む
def loadMeta(meta_dir, image_path):
    return np.load(f'{meta_dir}/{image_path}.npy')

#　画像のパスを取得
def getImageFiles(image_dir):
    dirPath = pathlib.Path(image_dir)
    files = []
    files.extend(dirPath.glob(f"**/*.png"))
    files.extend(dirPath.glob(f"**/*.jpg"))
    files.extend(dirPath.glob(f"**/*.gif"))
    files.extend(dirPath.glob(f"**/*.webp"))
    files = map(lambda f: str(f.relative_to(dirPath)), files)
    files = sorted(files)
    return files

#　メタデータのパスを取得
def getMetafilse(meta_dir):
    files = list(loadPickleFile(meta_dir).keys())
    files = sorted(files)
    return files
    

# モデルの読み込み
def loadModel(model_name=("ViT-L-14-336", "openai"), device="cpu"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1])

# テキストの埋め込みを計算
def encode_text(model, text):
    return model.encode_text(open_clip.tokenize(text)).to("cpu").detach().numpy().copy()

# 画像の埋め込みを計算
def encode_image(model, preprocess, image, device):
    image_input = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(
        image_input).to("cpu").detach().numpy().copy()
    return image_features
