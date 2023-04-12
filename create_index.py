import csv
import hashlib
import itertools
import json
import os
import pathlib
import sqlite3
import traceback
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


def create_meta_file(args, file, search_model, preprocess, device):
    try:
        data = Image.open(f'{args.image_dir}/{file}')
        image_input = preprocess(data).unsqueeze(0).to(device)
        image_features = search_model.encode_image(image_input).to("cpu").detach().numpy().copy().astype(np.float32)
        return image_features
    except:
        print(file)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="dir", default="./images")
    parser.add_argument("--meta_dir", help="dir", default="./meta")
    parser.add_argument("--search_model_name", help="model_name", default="ViT-L-14-336")
    parser.add_argument("--search_model_pretrained", help="pretrained", default="openai")
    parser.add_argument("--search_model_out_dim", help="search_model_out_dim", default=768)
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
    os.makedirs(f"{search_model_meta_dir}", exist_ok=True)
    con: sqlite3.Connection = sqlite3.connect(f"{search_model_meta_dir}/sqlite_image_meta.db", isolation_level="DEFERRED") #READ UNCOMMITTED
    cur: sqlite3.Cursor = con.cursor()

    cur.execute("pragma journal_mode = WAL")
    cur.execute("pragma synchronous = normal")
    cur.execute("pragma temp_store = memory")
    cur.execute("pragma mmap_size = 1073741824") # 1 GB

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
        jit=False
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

        amodel = get_aesthetic_model(clip_model="vit_l_14")
        amodel.eval()

        search_meta_list = []

        pbar = tqdm.tqdm(zip(result["valid_id"], result["meta"]), total=len(result["valid_id"]))
        for i, (image_id, meta) in enumerate(pbar):

            if meta != None:
                meta = np.squeeze(meta)
                a = np.frombuffer(meta, dtype=np.float32)
                if a.shape[0] == args.search_model_out_dim:
                    search_meta_list.append(a)
                    continue

            image_path = index_item_list[image_id]

            new_search_meta = create_meta_file(args, image_path, search_model, preprocess, device)

            if type(new_search_meta).__module__ != "numpy":
                continue
                
            if new_search_meta.shape[1] != args.search_model_out_dim:
                continue

            new_search_meta = np.squeeze(new_search_meta)
            search_meta_list.append(new_search_meta)
            new_search_meta_bytes = new_search_meta.tobytes()

            try:
                image_features = torch.from_numpy(new_search_meta)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                aesthetic_quality = amodel(image_features).item()
            except:
                aesthetic_quality: float = 0.0

            
            param = (image_id, image_path, new_search_meta_bytes, aesthetic_quality, image_path, new_search_meta_bytes, aesthetic_quality)
            try:
                cur.execute("insert into image_meta values(?, ?, ?, ?) on conflict(image_id) do update set image_path=?, meta=?, aesthetic_quality=?", param)
            except:
                traceback.print_exc()

            if i%10000 == 0:
                print("commit")
                con.commit()
            

        con.commit()
        print("CREATE INDEX IF NOT EXISTS idx_image_meta_image_id ON image_meta(image_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_image_meta_image_id ON image_meta(image_id)")
        con.commit()
        print("pragma vacuum")
        cur.execute("pragma vacuum")
        print("pragma optimize")
        cur.execute("pragma optimize")
        print("close")
        con.close()

        a = np.squeeze(np.array(search_meta_list, dtype=np.float32))
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
    main()