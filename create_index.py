import argparse
import hashlib
import itertools
import os
import pathlib
import sqlite3
import traceback
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Loader(Dataset):
    def __init__(self, batch_size, images_dir, file_names, transform):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.img_list = file_names
        self.transform = transform
        self.i = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        while True:
            try:
                image = Image.open(
                    f"{self.images_dir}/{self.img_list[self.i]}"
                ).convert("RGB")
                break
            except:
                traceback.print_exc()
                self.i += 1
                self.i %= len(self.img_list)

        image_index = self.i
        if self.transform is not None:
            image = self.transform(image)

        self.i += 1
        self.i %= len(self.img_list)
        return image, image_index


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="dir", default="./images")
    parser.add_argument("--meta_dir", help="dir", default="./meta")
    parser.add_argument(
        "--search_model_name", help="model_name", default="ViT-L-14-336"
    )
    parser.add_argument(
        "--search_model_pretrained", help="pretrained", default="openai"
    )
    parser.add_argument(
        "--search_model_out_dim", help="search_model_out_dim", default=768
    )
    parser.add_argument("--batch_size", help="batch size", default=64)
    parser.add_argument("--nlist", help="nlist", default=16)
    parser.add_argument("--M", help="M", default=256)
    parser.add_argument("--bits_per_code", help="bits_per_code", default=4)
    parser.add_argument(
        "--metas_faiss_index_file_name",
        help="metas_faiss_index_file_name",
        default="metafiles.index",
    )
    args = parser.parse_args()
    search_model_meta_dir: str = (
        f"{args.meta_dir}/{args.search_model_name}-{args.search_model_pretrained}"
    )

    print("files")
    dirPath: pathlib.Path = pathlib.Path(args.image_dir)
    print(dirPath)

    # 画像ファイルの一覧を作成
    extensions: list[str] = ["png", "jpg", "gif", "webp"]

    files = itertools.chain.from_iterable(dirPath.glob(f"**/*.{e}") for e in extensions)

    files_list: list[str] = sorted(map(lambda f: str(f.relative_to(dirPath)), files))

    print(len(files_list))

    print("load_db")
    os.makedirs(f"{search_model_meta_dir}", exist_ok=True)
    con: sqlite3.Connection = sqlite3.connect(
        f"{search_model_meta_dir}/sqlite_image_meta.db", isolation_level="DEFERRED"
    )  # READ UNCOMMITTED
    cur: sqlite3.Cursor = con.cursor()

    cur.execute("pragma journal_mode = WAL")
    cur.execute("pragma synchronous = normal")
    cur.execute("pragma temp_store = memory")
    cur.execute("pragma mmap_size = 1073741824")  # 1 GB

    cur.execute(
        """
                create table if not exists image_meta(
                  image_id text PRIMARY KEY,
                  image_path text,
                  meta blob,
                  aesthetic_quality real
                )"""
    )

    # GPUが使用可能か
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # モデルの読み込み
    print("load_model")

    search_model, _, eval_transform = open_clip.create_model_and_transforms(
        args.search_model_name,
        pretrained=args.search_model_pretrained,
        device=device,
        jit=False,
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
            placeholder: str = ",".join(["(?)"] * len(id_list[i : i + n]))
            result: pd.DataFrame = pd.read_sql_query(
                f"""

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

            """,
                con,
                params=id_list[i : i + n],
            )
            temp.append(result)

        result = pd.concat(temp)

        amodel = get_aesthetic_model(clip_model="vit_l_14")
        amodel.eval()

        search_meta_list = []
        uncreated_image_paths = []

        pbar = tqdm.tqdm(
            zip(result["valid_id"], result["meta"]), total=len(result["valid_id"])
        )
        for i, (image_id, meta) in enumerate(pbar):
            if meta is not None:
                meta = np.squeeze(meta)
                a = np.frombuffer(meta, dtype=np.float32)
                if a.shape[0] == args.search_model_out_dim:
                    search_meta_list.append(a)
                    continue

            uncreated_image_paths.append(index_item_list[image_id])

        print(
            f"meta is None:{len(uncreated_image_paths)}/{len(index_item_list)} ({len(uncreated_image_paths)/len(index_item_list)*100.:.4f}%)"
        )
        print(
            f"meta is not None:{len(search_meta_list)}/{len(index_item_list)} ({len(search_meta_list)/len(index_item_list)*100.:.4f}%)"
        )

        loader = DataLoader(
            Loader(
                args.batch_size,
                args.image_dir,
                uncreated_image_paths,
                transform=eval_transform,
            ),
            batch_size=64,
            shuffle=False,
            num_workers=1,
        )

        for i, (batched_image_input, batched_image_index) in enumerate(
            tqdm.tqdm(loader)
        ):
            batched_image_input = batched_image_input.to(device)
            try:
                batched_new_search_meta = (
                    search_model.encode_image(batched_image_input)
                    .to("cpu")
                    .detach()
                    .numpy()
                    .copy()
                    .astype(np.float32)
                )
            except:
                traceback.print_exc()
                continue

            for new_search_meta, image_index in zip(
                batched_new_search_meta, batched_image_index
            ):
                if type(new_search_meta).__module__ != "numpy":
                    continue

                if new_search_meta.shape[0] != args.search_model_out_dim:
                    continue

                search_meta_list.append(new_search_meta)
                new_search_meta_bytes = new_search_meta.tobytes()

                aesthetic_quality: float = 0.0
                try:
                    image_features = torch.from_numpy(new_search_meta)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    aesthetic_quality = amodel(image_features).item()
                except:
                    traceback.print_exc()

                image_path = uncreated_image_paths[image_index]
                image_id: str = hashlib.sha256(str(image_path).encode()).hexdigest()

                print(image_index)
                param = (
                    image_id,
                    image_path,
                    new_search_meta_bytes,
                    aesthetic_quality,
                    new_search_meta_bytes,
                    aesthetic_quality,
                )
                try:
                    cur.execute(
                        "insert into image_meta values(?, ?, ?, ?) on conflict(image_id) do update set meta=?, aesthetic_quality=?",
                        param,
                    )
                except:
                    traceback.print_exc()

            if (i * args.batch_size) % 10000 < args.batch_size:
                print("commit")
                con.commit()

        con.commit()
        print(
            "CREATE INDEX IF NOT EXISTS idx_image_meta_image_id ON image_meta(image_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_image_meta_image_id ON image_meta(image_id)"
        )
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
        index = faiss.IndexIVFPQ(
            q, dim, nlist, M, bits_per_code, faiss.METRIC_INNER_PRODUCT
        )
        print("train")
        index.train(a)
        print("add")
        index.add(a)
        print("write")
        faiss.write_index(
            index, f"{search_model_meta_dir}/{args.metas_faiss_index_file_name}"
        )


if __name__ == "__main__":
    main()
