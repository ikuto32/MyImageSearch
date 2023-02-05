import csv
import hashlib
import itertools
import json
import os
import pathlib
import shutil
import numpy as np
import torch
from PIL import Image
import tqdm
import argparse
import pickle
import open_clip
import faiss

metasPickleFile = "metafiles.pickle"
metasFaissIndexFile = "metafiles.index"

def loadMeta(meta_dir, image_path):
    return np.load(f'{meta_dir}/{image_path}.npy')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="dir", default="./images")
    parser.add_argument("--meta_dir", help="dir", default="./meta/ViT-B-32-laion2b_s34b_b79k/")
    parser.add_argument("--model_name", help="model_name", default="ViT-B-32")
    parser.add_argument("--pretrained", help="pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--nlist", help="nlist", default=64)
    parser.add_argument("--M", help="M", default=32)
    parser.add_argument("--bits_per_code", help="bits_per_code", default=4)
    parser.add_argument("--metas_pickle_file_name", help="metas_pickle_file_name", default="metafiles.pickle")
    parser.add_argument("--metas_faiss_index_file_name", help="metas_faiss_index_file_name", default="metafiles.index")
    args = parser.parse_args()
    print("load")
    
    metasPickleFile = args.metas_pickle_file_name
    metasFaissIndexFile = args.metas_faiss_index_file_name
    
    print("files")
    dirPath = pathlib.Path(args.image_dir)
    print(dirPath)

    #画像ファイルの一覧を作成
    extensions = ["png", "jpg", "gif", "webp"]
  
    files = itertools.chain.from_iterable(dirPath.glob(f'**/*.{e}') for e in extensions)

    files = map(lambda f: str(f.relative_to(dirPath)), files)
    files = sorted(files)


    # GPUが使用可能か
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # モデルの読み込み
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, device=device, pretrained=args.pretrained)
    
    print("mkdir")

    #ディレクトリを初期化
    # os.remove(args.meta_dir)
    # shutil.rmtree(args.meta_dir)
    # os.mkdir(args.meta_dir)
    
    with torch.no_grad():
        pbar = tqdm.tqdm(files)
        metas = {}
        for _, file in enumerate(pbar):
            filename = f'{file}'
            data = Image.open(f'{args.image_dir}/{file}')
            image_input = preprocess(data).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            try:
                os.makedirs(f'{args.meta_dir}/{os.path.split(file)[0]}')
            except:
                pass
            np.save(f'{args.meta_dir}/{file}.npy', image_features.to("cpu").detach().numpy().copy())
            metas[filename] = loadMeta(args.meta_dir, filename)

        with open(f'{args.meta_dir}/{metasPickleFile}', 'wb') as f:
            pickle.dump(metas, f)

        json_dict = {}
        for file in metas.keys():
            imege_id = hashlib.sha256(str(file).encode()).hexdigest()
            json_dict[imege_id] = file

        with open(f'{args.meta_dir}/item_json.json', 'w') as f:
            json.dump(json_dict, f)
        
        print("createIndexFile")
        metalist = list(metas.values())
        a = np.squeeze(np.asarray(metalist)).astype(np.float32)
        faiss.normalize_L2(a)
        dim = a.shape[1]
        q = faiss.IndexFlatIP(dim)
        nlist = args.nlist
        M = args.M
        bits_per_code = args.bits_per_code
        index = faiss.IndexIVFPQ(q, dim, nlist, M, bits_per_code, faiss.METRIC_INNER_PRODUCT)
        print("train")
        index.train(a)
        print("add")
        index.add(a)
        print("write")
        faiss.write_index(index, f'{args.meta_dir}/{metasFaissIndexFile}')
                

if __name__ == '__main__':
    main()