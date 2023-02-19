import csv
import hashlib
import itertools
import json
import os
import pathlib
import sqlite3
import numpy as np
import torch
from PIL import Image
import tqdm
import argparse
import pickle
import open_clip
import faiss
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
import asyncio
import pandas as pd


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


def create_meta_file(args, search_model_meta_dir, file, search_model, preprocess, device):
    
    meta_path: str = f'{search_model_meta_dir}/{file}.npy'

    if os.path.isfile(meta_path):
        try:
            return np.load(meta_path)
        except:
            pass
    try:
        data = Image.open(f'{args.image_dir}/{file}')
        image_input = preprocess(data).unsqueeze(0).to(device)
        image_features = search_model.encode_image(image_input).to("cpu").detach().numpy().copy().astype(np.float32)
        return image_features
    except:
        print(file)
        return None

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="dir", default="./images")
    parser.add_argument("--meta_dir", help="dir", default="F:/dataset/clip_meta")
    parser.add_argument("--search_model_name", help="model_name", default="ViT-L-14-336")
    parser.add_argument("--search_model_pretrained", help="pretrained", default="openai")
    parser.add_argument("--nlist", help="nlist", default=64)
    parser.add_argument("--M", help="M", default=128)
    parser.add_argument("--bits_per_code", help="bits_per_code", default=4)
    parser.add_argument("--metas_faiss_index_file_name", help="metas_faiss_index_file_name", default="metafiles.index")
    args = parser.parse_args()
    search_model_meta_dir: str = f"{args.meta_dir}/{args.search_model_name}-{args.search_model_pretrained}"
    
    metasFaissIndexFile = args.metas_faiss_index_file_name
    
    print("files")
    dirPath: pathlib.Path = pathlib.Path(args.image_dir)
    print(dirPath)

    #画像ファイルの一覧を作成
    extensions: list[str] = ["png", "jpg", "gif", "webp"]
  
    files = itertools.chain.from_iterable(dirPath.glob(f'**/*.{e}') for e in extensions)

    files_list: list[str]= sorted(map(lambda f: str(f.relative_to(dirPath)), files))

    print(len(files_list))

    print("load_db")
    con: sqlite3.Connection = sqlite3.connect(f"{search_model_meta_dir}/sqlite_image_meta.db", isolation_level="DEFERRED")
    cur: sqlite3.Cursor = con.cursor()
    cur.execute("""
                create table if not exists image_meta(
                  image_id text PRIMARY KEY,
                  image_path text,
                  meta blob,
                  aesthetic_quality real
                )""")

    # GPUが使用可能か
    device= "cuda" if torch.cuda.is_available() else "cpu"
    # モデルの読み込み
    print("load_model")

    search_model, _, preprocess = open_clip.create_model_and_transforms(
        args.search_model_name, 
        pretrained=args.search_model_pretrained,
        device=device,
        jit=True
    )
    
    with torch.no_grad():

        index_item_list: dict[str, str] = {}
        
        for file in tqdm.tqdm(files_list):
            image_id: str = hashlib.sha256(str(file).encode()).hexdigest()
            index_item_list[image_id] = file


        id_list: list[str] = list(index_item_list.keys())
        
        temp = []
        n = 10000
        for i in tqdm.tqdm(range(0, len(id_list), n)):
            placeholder: str = ",".join(["(?)"] * len(id_list[i:i + n]))
            result: pd.DataFrame = pd.read_sql_query(f"""

                /* 有効なIDの導出テーブル */
                WITH valid_id_table(valid_id) AS (
                VALUES 
                    {placeholder}
                )

                /* 有効なIDと突合する。 不足している場合、 その値は、nullになる。 */
                SELECT 
                    valid_id_table.valid_id,
                    image_meta.image_path,
                    image_meta.meta,
                    image_meta.aesthetic_quality
                FROM
                    valid_id_table
                    LEFT OUTER JOIN image_meta
                    ON valid_id_table.valid_id = image_meta.image_id

            """, con, params=id_list[i:i + n])
            temp.append(result)

        result = pd.concat(temp)

        amodel= get_aesthetic_model(clip_model="vit_l_14")
        amodel.eval()

        search_meta_list = []

        pbar = tqdm.tqdm(zip(result["valid_id"], result["meta"]), total=len(result["valid_id"]))
        for i, (image_id, meta) in enumerate(pbar):

            if meta != None:
                search_meta_list.append(np.frombuffer(meta, dtype=np.float32))
                continue

            image_path = index_item_list[image_id]

            new_search_meta = create_meta_file(args, search_model_meta_dir, image_path, search_model, preprocess, device)

            if type(new_search_meta).__module__ != "numpy":
                continue
                
            search_meta_list.append(new_search_meta)
            new_search_meta_bytes = new_search_meta.tobytes()

            quality_meta_path = f"{args.meta_dir}/ViT-L-14-336-openai/{image_path}.npy"
            try:
                new_quality_meta = np.load(quality_meta_path)
                image_features = torch.from_numpy(new_quality_meta)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                aesthetic_quality = amodel(image_features).item()
            except:
                aesthetic_quality: float = 0.0

            
            param = (image_id, image_path, new_search_meta_bytes, aesthetic_quality)
            cur.execute("insert into image_meta values(?, ?, ?, ?)", param)

            if i%1000 == 0:
                print("commit")
                con.commit()
            

        con.commit()
        con.close()
        

        a = np.squeeze(np.asarray(search_meta_list)).astype(np.float32)
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