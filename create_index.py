from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import datetime
import hashlib
import os
import pathlib
import pickle
import sqlite3
import traceback
import typing
from typing import Optional, Tuple, Union

import faiss
import huggingface_hub
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 268_435_456


def safe_collate(batch):
    # None（壊れ/変換失敗）を除外
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  # 上位ループでcontinue
    return default_collate(batch)


class DirImageIterable(IterableDataset):
    def __init__(self, images_dir: str, img_list, transform=None):
        self.images_dir = images_dir
        self.img_list = list(img_list)
        self.transform = transform

    def _open_rgb(self, path: str):
        try:
            with Image.open(path) as im:
                return im.convert("RGBA").convert("RGB").copy()
        except Exception as e:
            print(f"[DirImageIterable] open failed: {path} -> {e}")
            return None

    def __getitem__(self, index: int):
        """Iterable datasets do not support random access."""
        raise TypeError(f"{self.__class__.__name__} does not support indexing")

    def __iter__(self):
        try:
            info = get_worker_info()
            if info is None:
                it = range(len(self.img_list))
                worker_id = None
            else:
                # 各ワーカーに均等分割
                per_worker = (len(self.img_list) + info.num_workers - 1) // info.num_workers
                start = info.id * per_worker
                end = min(start + per_worker, len(self.img_list))
                it = range(start, end)
                print(f"[DirImageIterable] worker {info.id} processing {start} to {end} of {len(self.img_list)}")
                worker_id = info.id

                
            for idx in it:
                try:
                    rel = self.img_list[idx]
                except Exception as e:
                    print(f"[DirImageIterable] index failure: {idx} in {worker_id} -> {e}")
                    continue
                path: str = os.path.join(self.images_dir, rel)
                # print(f"[DirImageIterable] path: {path} ({idx}/{len(self.img_list)})")
                try:
                    img = self._open_rgb(path)
                except Exception as e:
                    # _open_rgb already prints details for known issues, but
                    # this extra guard keeps unexpected errors from crashing
                    # the worker process.
                    print(f"[DirImageIterable] unexpected open failure: {path} in {worker_id} -> {e}")
                    continue

                if img is None:
                    continue

                if self.transform is not None:
                    try:
                        img = self.transform(img)
                    except Exception as e:
                        print(f"[DirImageIterable] transform failed: {path} in {worker_id} -> {e}")
                        continue
                try:
                    yield img, idx
                except Exception as e:
                    print(f"[DirImageIterable] yield failed: {path} in {worker_id} -> {e}")
                    continue
        except Exception as e:
            traceback.print_exc()
            print(f"[DirImageIterable] unexpected iteration failure: {e} in {worker_id}")
            raise


class Aesthetic_model(nn.Module):
    def __init__(self, input_dim=768):
        super(Aesthetic_model, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024), nn.SiLU())
        self.dropout1 = nn.Dropout1d(0.5)
        self.fc2 = nn.Sequential(nn.Linear(1024, 1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


# ================================================
# 美的評価（Aesthetic Scorer）クラス
# ================================================
class PonyAestheticScorer:
    def __init__(self, device=None, checkpoint=None):
        """
        device: 使用するデバイス。未指定の場合、cuda:0が利用可能ならcuda、なければcpu
        checkpoint: 学習済みチェックポイントのパス。チェックポイントが存在すればモデル重みを読み込む
        """
        self.device = device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        # CLIPの出力次元は768を想定（実際のモデルに合わせて調整してください）
        self.aesthetic_model = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        ).to(self.device)
        if checkpoint is not None:
            try:
                checkpoint_data = torch.load(checkpoint, map_location=self.device)
                # checkpoint_dataの形式に応じてキーを取得（キー名が"state_dict"の場合など）
                state_dict = checkpoint_data.get('state_dict', checkpoint_data)
                # "model." のプレフィックスがあれば除去
                new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                self.aesthetic_model.load_state_dict(new_state_dict)
                print("Aesthetic model checkpoint loaded.")
            except Exception as e:
                print(f"Failed to load aesthetic checkpoint: {e}")
        self.aesthetic_model.eval()

    def score(self, features):
        """指定されたfeaturesから美的評価スコアを算出して返す"""
        # L2正規化
        norm = features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        features = features / norm
        with torch.no_grad():
            score = self.aesthetic_model(features).item()
        return score

# ================================================
# スタイルクラスタリング用クラス
# ================================================
class StyleCluster:
    def __init__(self, device=None, checkpoint_model=None, checkpoint_centers=None):
        """
        device: 使用するデバイス
        checkpoint_model: スタイルクラスタリング用モデルの学習済みチェックポイント（例：v3_checkpoint00120000.pth）
        checkpoint_centers: クラスタ中心の学習済みチェックポイント（例：centers_n1024.pkl）
        いずれも指定されなければ、CLIPモデルとランダム中心を用いる
        """
        self.device = device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 独自の前処理（参考実装と同様）
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        try:
            with open(checkpoint_centers, 'rb') as f:
                centers_data = pickle.load(f)
            original_centers = centers_data['cluster_centers']
            self.cluster_centers = torch.tensor(original_centers, dtype=torch.float32)
            # 正規化
            self.cluster_centers = self.cluster_centers / self.cluster_centers.norm(dim=1, keepdim=True)
            print("Style cluster centers loaded.")
        except Exception as e:
            print(f"Failed to load style cluster centers: {e}")
            self._init_random_centers()

    def _init_random_centers(self, num_clusters=10):
        torch.manual_seed(42)
        centers = torch.randn(num_clusters, 768)
        self.cluster_centers = centers / centers.norm(dim=1, keepdim=True)

    def get_cluster(self, features):
        """指定されたfeaturesからスタイルクラスタID（文字列）を返す"""
        centers = self.cluster_centers.to(self.device)
        distances = torch.norm(features - centers, dim=1)
        min_index = torch.argmin(distances).item()
        return str(min_index)


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
        "--aesthetic_model_path", help="aesthetic_model_path", default="./model/aesthetic_ranking100.pth"
    )
    parser.add_argument(
        "--search_model_name", help="model_name", default="ViT-SO400M-16-SigLIP-i18n-256"
    )
    parser.add_argument(
        "--search_model_pretrained", help="pretrained", default="webli"
    )
    parser.add_argument(
        "--search_model_out_dim", help="search_model_out_dim", type=int, default=1152
    )

    model_repo = "purplesmartai/aesthetic-classifier"
    MODEL_FILENAME = "v2.ckpt"
    parser.add_argument(
        "--aesthetic_checkpoint", help="aesthetic_checkpoint", default=huggingface_hub.hf_hub_download(model_repo, MODEL_FILENAME)
    )

    model_repo = "purplesmartai/style-classifier"
    MODEL_FILENAME = "v3_checkpoint00120000.pth"
    CENTERS_FILENAME = "clustering_results_n2048_gpu.npz"
    parser.add_argument(
        "--style_checkpoint_model", help="style_checkpoint_model", default=huggingface_hub.hf_hub_download(model_repo, MODEL_FILENAME)
    )
    parser.add_argument(
        "--style_checkpoint_centers", help="style_checkpoint_centers", default=huggingface_hub.hf_hub_download(model_repo, CENTERS_FILENAME)
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--num_workers", help="data loader workers", type=int, default=32)
    parser.add_argument("--nlist", help="centroid size", type=int, default=64)
    parser.add_argument("--M", help="M", type=int, default=1152)
    parser.add_argument("--bits_per_code", help="bits_per_code", type=int, default=8)
    parser.add_argument(
        "--metas_faiss_index_file_name",
        help="metas_faiss_index_file_name",
        default="metafiles.index",
    )

    return parser.parse_args()


_PARALLEL = 64

def get_image_list_from_dir(dir_path: str | os.PathLike, exts: typing.Sequence[str]) -> list[str]:
    """
    dir_path 以下を走査して、指定拡張子のファイルリスト（dir_path からの相対パス）を返す。
    SMB のレイテンシを考慮して並列数を抑えつつ、ディレクトリ単位でまとめてスレッドプールに流す。
    """
    return asyncio.run(_collect_images(pathlib.Path(dir_path), exts))


async def _collect_images(root: pathlib.Path, exts: typing.Sequence[str]) -> list[str]:
    loop = asyncio.get_running_loop()

    # .xyz と xyz の両方を許容
    ext_set = {("."+e if not e.startswith(".") else e).lower() for e in exts}

    files: list[str] = []
    bar = tqdm.tqdm(unit="file", dynamic_ncols=True)

    # 同時実行制限
    sem = asyncio.Semaphore(_PARALLEL)

    def _scan_dir_sync(path: pathlib.Path) -> tuple[list[pathlib.Path], list[str]]:
        """同期関数。path 配下を走査し、サブディレクトリとヒットしたファイルを返す。"""
        dirs, hits = [], []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    # SMB では d_type がないことがあるので try/except
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(pathlib.Path(entry.path))
                        else:
                            name_lc = entry.name.lower()
                            if any(name_lc.endswith(ext) for ext in ext_set):
                                hits.append(os.path.relpath(entry.path, root))
                    except OSError:
                        # アクセス不可は無視
                        pass
        except (PermissionError, FileNotFoundError):
            pass
        return dirs, hits

    async def _walk(path: pathlib.Path):
        async with sem:
            # thread pool で同期スキャン
            dirs, hits = await loop.run_in_executor(pool, _scan_dir_sync, path)
            files.extend(hits)
            bar.update(len(hits))

        # サブディレクトリを並列にたどる
        await asyncio.gather(*(_walk(d) for d in dirs))

    # プールは with で自動クローズ
    with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL) as pool:
        await _walk(root)

    bar.close()
    return sorted(files)


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
            pony_aesthetic_quality real,
            image_tags text,
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


def extract_image_features(
    args,
    device,
    search_model,
    aesthetic_model,
    con,
    cur,
    loader,
    uncreated_image_paths,
):
    # 事前学習済みチェックポイントのパスは args から取得（各自適宜設定してください）
    aesthetic_checkpoint = getattr(args, "aesthetic_checkpoint", None)
    style_checkpoint_model = getattr(args, "style_checkpoint_model", None)
    style_checkpoint_centers = getattr(args, "style_checkpoint_centers", None)

    # 美的評価（PonyAestheticScorer）とスタイルクラスタリング（StyleCluster）のグローバルインスタンスを生成
    pony_scorer = PonyAestheticScorer(device=device, checkpoint=aesthetic_checkpoint)
    style_cluster = StyleCluster(device=device, checkpoint_model=style_checkpoint_model, checkpoint_centers=style_checkpoint_centers)
    processed_count = 0

    for batch in tqdm.tqdm(loader, total=len(uncreated_image_paths) // args.batch_size + 1):
        if batch is None:
            continue

        batched_image_input, batched_image_index = batch

        batched_image_input = batched_image_input.to(device, non_blocking=True)
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

        for new_search_meta, image_index in zip(batched_new_search_meta, batched_image_index):
            if type(new_search_meta).__module__ != "numpy":
                print(f"{type(new_search_meta).__module__} != numpy")
                continue

            if new_search_meta.shape[0] != args.search_model_out_dim:
                print(f"{new_search_meta.shape[0]} != {args.search_model_out_dim}")
                continue

            new_search_meta_bytes = new_search_meta.tobytes()

            aesthetic_quality: float = 0.0
            pony_aesthetic_quality: float = 0.0
            image_tags: str = ''

            if new_search_meta.shape[0] == 768:
                try:
                    image_features = torch.from_numpy(new_search_meta)
                    denom = image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    image_features = (image_features / denom).contiguous()
                    aesthetic_quality = torch.sigmoid(
                        aesthetic_model(image_features.unsqueeze(0))
                    ).item()
                except Exception:
                    traceback.print_exc()

                new_search_meta_tensor: Optional[torch.Tensor] = None
                try:
                    new_search_meta_tensor = torch.from_numpy(new_search_meta).to(device).unsqueeze(0)
                    pony_aesthetic_quality = pony_scorer.score(new_search_meta_tensor)
                except Exception:
                    traceback.print_exc()
                    pony_aesthetic_quality = 0.0

                if new_search_meta_tensor is not None:
                    try:
                        style_cluster_number = style_cluster.get_cluster(new_search_meta_tensor)
                        image_tags = f"style_cluster_{style_cluster_number}"
                    except Exception:
                        traceback.print_exc()
                        image_tags = ''

            image_index_int = int(image_index)
            image_path = uncreated_image_paths[image_index_int]
            image_id: str = hashlib.sha256(str(image_path).encode()).hexdigest()

            # UTC time
            time_stamp_ISO = datetime.datetime.now(datetime.timezone.utc).isoformat()

            param = (
                image_id,
                image_path,
                new_search_meta_bytes,
                aesthetic_quality,
                pony_aesthetic_quality,
                image_tags,
                time_stamp_ISO,
                new_search_meta_bytes,
                aesthetic_quality,
                pony_aesthetic_quality,
                image_tags,
                time_stamp_ISO,
            )
            try:
                cur.execute(
                    "insert into image_meta values(?, ?, ?, ?, ?, ?, ?) on conflict(image_id) do update set meta=?, aesthetic_quality=?, pony_aesthetic_quality=?, image_tags=?, time_stamp_ISO=?",
                    param,
                )
                processed_count += 1
            except Exception:
                traceback.print_exc()

        if (processed_count) % 10000 < args.batch_size:
            print("commit")
            con.commit()


def collect_train_samples_algL(con: sqlite3.Connection, d: int, train_samples: int, batch_size: int):
    """
    Algorithm L (Vitter 1985) による reservoir sampling（読み飛ばし式）。
    image_meta(meta BLOB, image_path TEXT) を image_path ORDER BY で走査し、
    次元 d の float32 ベクトルを k=train_samples 件だけ均一サンプルする。
    """
    k = train_samples
    # 1) リザーバを確保
    reservoir = np.empty((k, d), dtype=np.float32)
    filled = 0

    # ORDER BY は前回答と同じ
    cur = con.execute("""
        SELECT meta
          FROM image_meta
         WHERE meta IS NOT NULL
         ORDER BY image_path
    """)

    # --- フェーズA: 先頭 k 件でリザーバを満たす ---
    while filled < k:
        rows = cur.fetchmany(batch_size)
        if not rows:
            # 総件数 < k の場合はここで打ち切り（そのまま返す）
            return reservoir[:filled]
        for (meta_blob,) in rows:
            a = np.frombuffer(meta_blob, dtype=np.float32)
            if a.size != d:
                continue
            reservoir[filled] = a
            filled += 1
            if filled >= k:
                break

    # --- フェーズB: Algorithm L で読み飛ばし ---
    # W の初期化（U in (0,1)）
    rng = np.random.default_rng(42)
    W = np.exp(np.log(rng.random()) / k)  # 0 < W < 1

    # 現在位置 i（1始まりとみなす）。すでに k 件消費済みなので i = k
    i = k

    # rows バッファを使い回す
    rows = cur.fetchmany(batch_size)
    row_idx = 0
    # rows を使い切ったら次を fetchmany するヘルパ
    def next_row():
        nonlocal rows, row_idx
        if row_idx >= len(rows):
            rows = cur.fetchmany(batch_size)
            row_idx = 0
            if not rows:
                return None
        item = rows[row_idx]
        row_idx += 1
        return item

    while True:
        # (1) スキップ長 g を生成
        #    g = floor( ln(U) / ln(1 - W) )
        U = rng.random()
        # ln(1 - W) は負、ln(U) も負なので商は正
        g = int(np.floor(np.log(U) / np.log(1.0 - W)))
        # g 件を読み飛ばす
        skipped = 0
        while skipped < g:
            rec = next_row()
            if rec is None:
                # データ終端
                return reservoir
            # meta の長さチェックなどは不要（読み飛ばし）
            i += 1
            skipped += 1

        # (2) 読み飛ばし後の1件を取り出し、これを候補にする
        rec = next_row()
        if rec is None:
            return reservoir
        (meta_blob,) = rec
        a = np.frombuffer(meta_blob, dtype=np.float32)
        i += 1  # この1件を消費
        if a.size == d:
            # リザーバのランダム位置を置換
            j = rng.integers(0, k)
            reservoir[j, :] = a

        # (3) W を更新： W <- W * exp( ln(U') / k )
        U_prime = rng.random()
        W = W * np.exp(np.log(U_prime) / k)

    # 到達しない


def stream_build_faiss(
    db_path: str,
    nlist: int, M: int, bits_per_code: int,
    d: int,                       # args.search_model_out_dim
    out_index_path: str,
    batch_size: int = 50_000,     # メモリに合わせて調整
    train_samples: int = 2_000_000, # 訓練に使う最大サンプル数（d次元×この件数だけ常駐）
):
    con = connect_db(db_path)
    try:
        con.execute("PRAGMA case_sensitive_like=OFF")
        con.execute("PRAGMA temp_store=MEMORY")  # ソートの一部をメモリに。ただし無理はしない
        # image_pathでのORDER BYを効かせるために、可能なら事前にINDEXを作っておく:
        # CREATE INDEX IF NOT EXISTS idx_image_meta_path ON image_meta(image_path);

        # 件数
        total = con.execute("SELECT COUNT(*) FROM image_meta").fetchone()[0]
        print(f"total rows: {total}")

        # 正規化して訓練
        train_buf = collect_train_samples_algL(con, d, train_samples, batch_size)
        faiss.normalize_L2(train_buf)
        index = createIndex(nlist, M, bits_per_code, d)  # 既存関数を利用
        print(f"train {train_buf.shape}")
        index.train(train_buf)
        del train_buf  # 訓練後は解放

        # -------- パス2: 逐次add（バッチ正規化→add） --------
        print("add")
        cur = con.execute("""
            SELECT image_path, meta
              FROM image_meta
             WHERE meta IS NOT NULL
             ORDER BY image_path
        """)

        pbar = tqdm.tqdm(total=total, unit="row")
        buf = np.empty((batch_size, d), dtype=np.float32)
        k = 0

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                # 端数のflush
                if k > 0:
                    faiss.normalize_L2(buf[:k])
                    index.add(buf[:k])
                break

            k = 0
            for (_, meta_blob) in rows:
                a = np.frombuffer(meta_blob, dtype=np.float32)
                if a.size != d:
                    pbar.update(1)
                    continue
                buf[k, :] = a
                k += 1
                pbar.update(1)

            if k > 0:
                faiss.normalize_L2(buf[:k])
                index.add(buf[:k])

        pbar.close()

        print("write")
        faiss.write_index(index, out_index_path)
    finally:
        con.close()


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

    # ext_list = list(Image.registered_extensions().keys())
    ext_list = ['.avif', '.avifs', '.blp', '.bmp', '.dib', '.bufr', '.cur', '.pcx', '.dcx', '.dds', '.ps', '.eps', '.fit', '.fits', '.fli', '.flc', '.ftc', '.ftu', '.gbr', '.gif', '.grib', '.h5', '.hdf', '.png', '.apng', '.jp2', '.j2k', '.jpc', '.jpf', '.jpx', '.j2c', '.icns', '.ico', '.im', '.iim', '.jfif', '.jpe', '.jpg', '.jpeg', '.mpg', '.mpeg', '.tif', '.tiff', '.mpo', '.msp', '.palm', '.pcd', '.pxr', '.pbm', '.pgm', '.ppm', '.pnm', '.pfm', '.psd', '.qoi', '.bw', '.rgb', '.rgba', '.sgi', '.ras', '.tga', '.icb', '.vda', '.vst', '.webp', '.wmf', '.emf', '.xbm', '.xpm']

    print(ext_list)

    index_item_list = get_image_list_from_dir(dir_path, exts=ext_list)
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

    del result, index_item_list, search_meta_list

    dataset = DirImageIterable(
        args.image_dir,
        uncreated_image_paths,
        transform=eval_transform,
    )
    T = True
    i = 0
    loader = None
    if not uncreated_image_paths:
        print("No new images to process.")
    else:
        num_workers = max(0, args.num_workers)
        dataloader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=safe_collate,
        )
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 16  # default=2

    while T:
        loader = DataLoader(
            dataset,
            **dataloader_kwargs,
        )
        print(f"try {i}")
        try:
            extract_image_features(
                args,
                device,
                search_model,
                aesthetic_model,
                con,
                cur,
                loader,
                uncreated_image_paths[i * args.batch_size:],
            )
            T = False
        except Exception:
            traceback.print_exc()
            dataset = DirImageIterable(
                args.image_dir,
                uncreated_image_paths[i * args.batch_size:],
                transform=eval_transform,
            )
            i += 1
            continue

    con.commit()

    del uncreated_image_paths, aesthetic_model, search_model, dataset, loader
    torch.cuda.empty_cache()

    print(
        "CREATE INDEX IF NOT EXISTS idx_image_meta_image_id ON image_meta(image_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_meta_image_id ON image_meta(image_id)"
    )
    con.commit()
    print(
        "CREATE INDEX IF NOT EXISTS idx_image_meta_path ON image_meta(image_path)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_meta_path ON image_meta(image_path)"
    )
    con.commit()
    print("pragma vacuum")
    cur.execute("pragma vacuum")
    print("pragma optimize")
    cur.execute("pragma optimize")
    print("close")
    con.close()

    del con, cur

    stream_build_faiss(search_model_meta_dir, args.nlist, args.M, args.bits_per_code, args.search_model_out_dim,
        f"{search_model_meta_dir}/{args.metas_faiss_index_file_name}", batch_size=100_000, train_samples=2_000_000
    )


if __name__ == "__main__":
    main()
