import argparse
import datetime
import hashlib
import itertools
import os
import pathlib
import sqlite3
import traceback

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, get_worker_info

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DirImageDataset(Dataset):
    """Custom Dataset for loading images from a directory."""

    def __init__(self, images_dir, img_list, transform):
        self.images_dir = images_dir
        self.img_list = img_list
        self.transform = transform
        self.i = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, _):

        image_path = None
        image = None

        for _ in self.img_list:
            try:
                image_path = f"{self.images_dir}/{self.img_list[self.i]}"
                image = Image.open(image_path).convert("RGB")
                break
            except Exception as e:
                print(f"An error occurred while opening image: {image_path}")
                print(e)
                self.i += 1
                self.i %= len(self.img_list)

        assert image_path is not None or image is not None, "An error occurred while opening image"

        image_index = self.i
        if self.transform is not None:
            image = self.transform(image)

        self.i += 1
        self.i %= len(self.img_list)
        return image, image_index


def worker_init_fn(worker_id):
    """Initializes dataset workers for DataLoader."""

    worker_info = get_worker_info()
    dataset = worker_info.dataset  # type: ignore the dataset copy in this worker process
    print(worker_info)
    sub_dataset_size = (len(dataset.img_list) // worker_info.num_workers)  # type: ignore
    dataset.i = sub_dataset_size * worker_id  # type: ignore
    print(f"sub dataset size:{sub_dataset_size}")
    print(f"dataset index:{dataset.i}")  # type: ignore


class Aesthetic_model(nn.Module):
    def __init__(self, input_dim=768):
        super(Aesthetic_model, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024), nn.SiLU())
        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.SiLU(), nn.Dropout())
        self.fc3 = nn.Sequential(nn.Linear(1024, 1024), nn.SiLU(), nn.Dropout())
        self.fc4 = nn.Sequential(nn.Linear(1024, 1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def get_aesthetic_model(path_to_model, clip_model="vit_l_14"):
    """load the aethetic model"""
    if clip_model == "vit_l_14":
        m = Aesthetic_model()
    else:
        raise ValueError()
    m.load_state_dict(torch.load(path_to_model))
    m.eval()
    return m


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="dir", default="./images")
    parser.add_argument("--meta_dir", help="dir", default="./clip_meta")
    parser.add_argument(
        "--aesthetic_model_path", help="aesthetic_model_path", default="./model/aesthetic_ranking20.pth"
    )
    parser.add_argument(
        "--search_model_name", help="model_name", default="ViT-L-14-336"
    )
    parser.add_argument(
        "--search_model_pretrained", help="pretrained", default="openai"
    )
    parser.add_argument(
        "--search_model_out_dim", help="search_model_out_dim", default=768
    )
    parser.add_argument("--batch_size", help="batch size", default=256)
    parser.add_argument("--nlist", help="centroid size", default=64)
    parser.add_argument("--M", help="M", default=768)
    parser.add_argument("--bits_per_code", help="bits_per_code", default=8)
    parser.add_argument(
        "--metas_faiss_index_file_name",
        help="metas_faiss_index_file_name",
        default="metafiles.index",
    )

    return parser.parse_args()


def get_image_list_from_dir(dir_path: pathlib.Path, ext: list[str]) -> list[str]:

    # 画像ファイルの一覧を作成
    files = itertools.chain.from_iterable(dir_path.glob(f"**/*.{e}") for e in ext)

    file_list: list[str] = sorted(map(lambda f: str(f.relative_to(dir_path)), tqdm.tqdm(files)))

    return file_list


def connect_db(current_dir: str) -> sqlite3.Connection:

    con: sqlite3.Connection = sqlite3.connect(
        f"{current_dir}/sqlite_image_meta.db", isolation_level="DEFERRED"
    )  # READ UNCOMMITTED

    return con


def init_db(cur: sqlite3.Cursor):

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
            aesthetic_quality real,
            time_stamp_ISO text
        )
        """
    )


def load_image_meta_from_db(con: sqlite3.Connection, id_list: list[str], loop_size: int) -> pd.DataFrame:

    temp: list[pd.DataFrame] = []

    # 指定サイズごとでループする。
    for idx in tqdm.tqdm(range(0, len(id_list), loop_size)):

        # 指定サイズのリスト
        sub_list: list[str] = id_list[idx: idx + loop_size]

        # データベースに問い合わせる
        temp.append(pd.read_sql_query(
            f"""

            /* 有効なIDの導出テーブル(IDの配列) */
            WITH valid_id_table(valid_id) AS (
            VALUES
                {",".join(["(?)"] * len(sub_list))}
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
            params=sub_list  # type: ignore
        ))

    return pd.concat(temp)


def createIndex(n_centroids, M, bits_per_code, dim) -> faiss.IndexIVFPQ:

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, n_centroids, M, bits_per_code, faiss.METRIC_INNER_PRODUCT
    )

    return index


def create_search_model_meta_dir(args):
    search_model_meta_dir = f"{args.meta_dir}/{args.search_model_name}-{args.search_model_pretrained}"
    os.makedirs(search_model_meta_dir, exist_ok=True)
    return search_model_meta_dir


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_clip_model(args, device):
    search_model, _, eval_transform = open_clip.create_model_and_transforms(
        args.search_model_name,
        pretrained=args.search_model_pretrained,
        device=device,
        jit=False,
    )
    return search_model, eval_transform


def create_image_id_index(file_list):
    index_item_list = {}
    for file in file_list:
        image_id = hashlib.sha256(str(file).encode()).hexdigest()
        index_item_list[image_id] = file
    return index_item_list


def filter_valid_image_meta(args, result, index_item_list):
    search_meta_list = []
    uncreated_image_paths = []
    for i, (image_id, meta) in enumerate(tqdm.tqdm(zip(result["valid_id"], result["meta"]), total=len(result["valid_id"]))):
        if meta is not None:
            meta = np.squeeze(meta)
            a = np.frombuffer(meta, dtype=np.float32)
            if a.shape[0] == args.search_model_out_dim:
                search_meta_list.append(a)
                continue

        uncreated_image_paths.append(index_item_list[image_id])
    return search_meta_list, uncreated_image_paths


def print_image_meta_stats(search_meta_list, uncreated_image_paths, max_len):
    print(
        f"uncreated images:{len(uncreated_image_paths)}/{max_len} ({len(uncreated_image_paths)/max_len*100.:.4f}%)"
    )
    print(
        f"existing metas:{len(search_meta_list)}/{max_len} ({len(search_meta_list)/max_len*100.:.4f}%)"
    )


def extract_image_features(args, device, search_model, aesthetic_model, con, cur, loader, search_meta_list, uncreated_image_paths):
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
        except Exception:
            traceback.print_exc()
            continue

        for new_search_meta, image_index in zip(
            batched_new_search_meta, batched_image_index
        ):
            if type(new_search_meta).__module__ != "numpy":
                print(f"{type(new_search_meta).__module__} != numpy")
                continue

            if new_search_meta.shape[0] != args.search_model_out_dim:
                print(f"{new_search_meta.shape[0]} != {args.search_model_out_dim}")
                continue

            search_meta_list.append(new_search_meta)
            new_search_meta_bytes = new_search_meta.tobytes()

            aesthetic_quality: float = 0.0
            if new_search_meta.shape[0] == 768:
                try:
                    image_features = torch.from_numpy(new_search_meta)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    aesthetic_quality = torch.sigmoid(aesthetic_model(image_features)).item()
                except Exception:
                    traceback.print_exc()

            image_path = uncreated_image_paths[image_index]
            image_id: str = hashlib.sha256(str(image_path).encode()).hexdigest()

            # UTC time
            time_stamp_ISO = datetime.datetime.now(datetime.timezone.utc).isoformat()

            param = (
                image_id,
                image_path,
                new_search_meta_bytes,
                aesthetic_quality,
                time_stamp_ISO,

                new_search_meta_bytes,
                aesthetic_quality,
                time_stamp_ISO
            )
            try:
                cur.execute(
                    "insert into image_meta values(?, ?, ?, ?, ?) on conflict(image_id) do update set meta=?, aesthetic_quality=?, time_stamp_ISO=?",
                    param,
                )
            except Exception:
                traceback.print_exc()

        if (i * args.batch_size) % 10000 < args.batch_size:
            print("commit")
            con.commit()


@torch.no_grad()
def main():

    # 起動引数
    args = parse_arguments()

    # 作業フォルダの作成
    search_model_meta_dir: str = create_search_model_meta_dir(args)

    # 画像ファイル一覧を求める
    print("files")
    dir_path: pathlib.Path = pathlib.Path(args.image_dir)
    print(dir_path)

    index_item_list = get_image_list_from_dir(dir_path, ext=["png", "jpg", "gif", "webp"])
    print(len(index_item_list))

    # データベースの準備
    print("load_db")
    con: sqlite3.Connection = connect_db(search_model_meta_dir)
    cur: sqlite3.Cursor = con.cursor()
    init_db(cur)

    # GPUが使用可能か
    device = get_device()

    # モデルの読み込み
    print("load_model")

    search_model, eval_transform = load_clip_model(args, device)

    aesthetic_model = get_aesthetic_model(args.aesthetic_model_path, clip_model="vit_l_14")

    index_item_list = create_image_id_index(index_item_list)

    # データベースに問い合わせる
    result = load_image_meta_from_db(con, id_list=list(index_item_list.keys()), loop_size=10000)

    search_meta_list, uncreated_image_paths = filter_valid_image_meta(args, result, index_item_list)

    print_image_meta_stats(search_meta_list, uncreated_image_paths, len(index_item_list))

    del result, index_item_list

    loader = DataLoader(
        DirImageDataset(
            args.image_dir,
            uncreated_image_paths,
            transform=eval_transform,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    extract_image_features(args, device, search_model, aesthetic_model, con, cur, loader, search_meta_list, uncreated_image_paths)

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

    del uncreated_image_paths, aesthetic_model, search_model, loader, con, cur
    # データベースに問い合わせる。
    con: sqlite3.Connection = connect_db(search_model_meta_dir)
    result = pd.read_sql_query(
        """
        SELECT image_id, image_path, meta FROM image_meta
        """, con)
    sorted_result = result.sort_values('image_path').reset_index()

    del result

    con.close()

    pbar = tqdm.tqdm(
        zip(sorted_result["image_path"], sorted_result["meta"]), total=len(sorted_result["image_path"])
    )

    del sorted_result

    search_meta_list = []
    for (_, meta) in pbar:
        if meta is not None:
            meta = np.squeeze(meta)
            a = np.frombuffer(meta, dtype=np.float32)
            if a.shape[0] == args.search_model_out_dim:
                search_meta_list.append(a)

    del pbar
    # faissの入力形式に変換する。
    a = np.squeeze(np.array(search_meta_list, dtype=np.float32))

    del search_meta_list

    # 正規化する。
    faiss.normalize_L2(a)

    # indexの作成
    index = createIndex(args.nlist, args.M, args.bits_per_code, a.shape[1])

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