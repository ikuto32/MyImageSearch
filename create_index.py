import csv
import hashlib
import itertools
import json
from multiprocessing import Pool
import os
import pathlib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import argparse
import pickle
import open_clip
import faiss
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
import asyncio


metasPickleFile = "metafiles.pickle"
metasFaissIndexFile = "metafiles.index"


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


async def create_meta_file(args, search_model_meta_dir, file, search_model, preprocess, device):
    meta_path = f'{search_model_meta_dir}/{file}.npy'
    if os.path.isfile(meta_path):
        try:
            return file, np.load(meta_path)
        except:
            pass
    try:
        data = Image.open(f'{args.image_dir}/{file}')
        image_input = preprocess(data).unsqueeze(0).to(device)
        image_features = search_model.encode_image(image_input).to("cpu").detach().numpy().copy()
        os.makedirs(f'{search_model_meta_dir}/{os.path.split(file)[0]}', exist_ok=True)
        np.save(f'{search_model_meta_dir}/{file}.npy', image_features)
        return file, image_features
    except:
        print(file)
        return None, None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="dir", default="./images")
    parser.add_argument("--meta_dir", help="dir", default="./meta")
    parser.add_argument("--search_model_name", help="model_name", default="ViT-L-14-336")
    parser.add_argument("--search_model_pretrained", help="pretrained", default="openai")
    parser.add_argument("--nlist", help="nlist", default=16)
    parser.add_argument("--M", help="M", default=256)
    parser.add_argument("--bits_per_code", help="bits_per_code", default=4)
    parser.add_argument("--metas_pickle_file_name", help="metas_pickle_file_name", default="metafiles.pickle")
    parser.add_argument("--metas_faiss_index_file_name", help="metas_faiss_index_file_name", default="metafiles.index")
    args = parser.parse_args()
    search_model_meta_dir: str = f"{args.meta_dir}/{args.search_model_name}-{args.search_model_pretrained}"
    print("load")
    
    metasPickleFile = args.metas_pickle_file_name
    metasFaissIndexFile = args.metas_faiss_index_file_name
    
    print("files")
    dirPath: pathlib.Path = pathlib.Path(args.image_dir)
    print(dirPath)

    #画像ファイルの一覧を作成
    extensions: list[str] = ["png", "jpg", "gif", "webp"]
  
    files = itertools.chain.from_iterable(dirPath.glob(f'**/*.{e}') for e in extensions)

    files_list: list[str]= sorted(map(lambda f: str(f.relative_to(dirPath)), files))


    # GPUが使用可能か
    device= "cuda" if torch.cuda.is_available() else "cpu"
    # モデルの読み込み

    search_model, _, preprocess = open_clip.create_model_and_transforms(
        args.search_model_name, 
        pretrained=args.search_model_pretrained,
        device=device,
        jit=True
    )

    print("mkdir")

    #ディレクトリを初期化
    # os.remove(args.meta_dir)
    # shutil.rmtree(args.meta_dir)
    # os.mkdir(args.meta_dir)
    
    with torch.no_grad():
        pbar: tqdm[str] = tqdm(files_list)
        index_item_list = {}
        metas = {}

        tasks = []
        metas = {}
        for file in pbar:
            tasks.append(asyncio.create_task(create_meta_file(args, search_model_meta_dir, file, search_model, preprocess, device)))
            await asyncio.sleep(0.0001)
        
        for i, _ in enumerate(pbar):
            file, meta = await tasks[i]
            if file:
                metas[file] = meta


        for file in tqdm(metas.keys()):
            imege_id: str = hashlib.sha256(str(file).encode()).hexdigest()
            index_item_list[imege_id] = {"path":file}
        
        with open(f'{search_model_meta_dir}/{metasPickleFile}', 'wb') as f:
            pickle.dump(metas, f)

        with open(f'{args.meta_dir}/index_item_list.json', 'w') as f:
            json.dump(index_item_list, f)
        

        aesthetic_quality_json_dict = {}

        amodel= get_aesthetic_model(clip_model="vit_l_14")
        amodel.eval()

        for file in tqdm(metas.keys()):
            try:
                meta_path = f"{args.meta_dir}/ViT-L-14-336-openai/{file}.npy"

                image_features = torch.from_numpy(np.load(meta_path))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                aesthetic_quality = amodel(image_features).item()
            except:
                aesthetic_quality: float = 0.0
            imege_id: str = hashlib.sha256(str(file).encode()).hexdigest()
            aesthetic_quality_json_dict[imege_id] = {"path":file,"aesthetic_quality": aesthetic_quality}

        with open(f'{args.meta_dir}/aesthetic_quality.json', 'w') as f:
            json.dump(aesthetic_quality_json_dict, f)
        
        print("createIndexFile")
        metalist = list(metas.values())
        a = np.squeeze(np.asarray(metalist)).astype(np.float32)
        faiss.normalize_L2(a)
        dim: int = a.shape[1]
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
        faiss.write_index(index, f'{search_model_meta_dir}/{metasFaissIndexFile}')
                

if __name__ == '__main__':
    asyncio.run(main())