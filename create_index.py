from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import datetime
import hashlib
import json
import os
import pathlib
import sqlite3
import traceback
import typing
from typing import Any, Iterable
import csv
import uuid

import faiss
import huggingface_hub
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate
import onnxruntime as rt


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 268_435_456


# =========================
# 共通ヘルパー
# =========================

def safe_collate(batch):
    """Custom collate that also carries raw PIL images."""

    # None（壊れ/変換失敗）を除外
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  # 上位ループでcontinue

    image_tensors = [b[0] for b in batch]
    indices = [b[1] for b in batch]
    raw_images = [b[2] for b in batch]

    return default_collate(image_tensors), default_collate(indices), raw_images


AESTHETIC_REPO = "purplesmartai/aesthetic-classifier"
AESTHETIC_CHECKPOINT_FILENAME = "v2.ckpt"
STYLE_REPO = "purplesmartai/style-classifier"
STYLE_MODEL_FILENAME = "v3_checkpoint00120000.pth"
STYLE_CENTERS_FILENAME = "clustering_results_n2048_gpu.npz"


def ensure_asset_path(asset_arg: str | None, filename: str, description: str) -> str:
    """Return a local path for an asset, downloading from HF if needed."""

    if asset_arg is None:
        raise FileNotFoundError(
            f"No {description} provided. Supply a local path or a Hugging Face repo ID."
        )

    if os.path.exists(asset_arg):
        return asset_arg

    try:
        return huggingface_hub.hf_hub_download(asset_arg, filename)
    except Exception as exc:  # pragma: no cover - user input/dependency failure
        raise FileNotFoundError(
            f"Could not resolve {description} from '{asset_arg}'. "
            "Provide an existing file path or a valid Hugging Face repo ID."
        ) from exc


kaomojis = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"
]

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


def load_labels(dataframe: pd.DataFrame):
    """wd-tagger 用のラベル読み込み."""
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def mcut_threshold(probs: np.ndarray) -> float:
    """最小カットっぽい閾値計算."""
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return float(thresh)


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

                raw_image = img.copy()

                if self.transform is not None:
                    try:
                        img = self.transform(img)
                    except Exception as e:
                        print(f"[DirImageIterable] transform failed: {path} in {worker_id} -> {e}")
                        continue
                try:
                    yield img, idx, raw_image
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
    def __init__(self, device='cuda:0', input_size=768, checkpoint=None):
        """
        device: 使用するデバイス。未指定の場合、cuda:0が利用可能ならcuda、なければcpu
        checkpoint: 学習済みチェックポイントのパス。チェックポイントが存在すればモデル重みを読み込む
        """
        self.device = device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        # CLIPの出力次元は768を想定（実際のモデルに合わせて調整してください）
        self.aesthetic_model = nn.Sequential(
            nn.Linear(input_size, 1024),
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

    @torch.no_grad()
    def score(self, features):
        """指定されたfeaturesから美的評価スコアを算出して返す"""
        # L2正規化
        norm = features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        norm[norm == 0] = 1
        features = features / norm
        with torch.no_grad():
            aesthetic_score = self.aesthetic_model(features).item()
        return aesthetic_score

    @torch.no_grad()
    def score_batch(self, features: torch.Tensor) -> np.ndarray:
        """
        バッチ版:
        features: (N, D) on self.device or cpu
        return: (N,) の np.ndarray[float]
        """
        if not torch.is_tensor(features):
            features = torch.from_numpy(features)

        features = features.to(self.device)
        # L2正規化
        norm = features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        norm[norm == 0] = 1
        features = features / norm

        with torch.no_grad():
            out = self.aesthetic_model(features)  # (N,1)
        return out.squeeze(-1).cpu().numpy()


# ================================================
# スタイルクラスタリング用クラス
# ================================================

# ブロックリスト（不要なクラスタを除外）
BLOCKLIST_CLUSTERS = set([
    1537, 1540, 1544, 520, 532, 1046, 24, 26, 1051, 1566, 31,
    545, 1058, 1572, 1061, 1575, 1065, 1066, 1580, 1072, 2045,
    560, 1591, 1085, 1598, 1599, 62, 1602, 1604, 1606, 584,
    1612, 81, 1111, 1626, 1115, 1632, 1634, 610, 616, 1642,
    620, 114, 627, 633, 1148, 640, 647, 138, 1675, 1676, 670,
    159, 1185, 161, 1704, 1198, 686, 687, 180, 1205, 692, 1208,
    1211, 188, 1725, 700, 1727, 1730, 1224, 1737, 1226, 1229,
    1742, 1743, 209, 1235, 1749, 725, 224, 737, 230, 231, 1258,
    1770, 747, 1261, 1273, 1785, 1281, 1793, 1286, 1806, 271,
    1299, 275, 276, 1816, 796, 797, 288, 804, 1321, 811, 1837,
    302, 816, 818, 1850, 1341, 1350, 1863, 329, 1866, 1355,
    1872, 851, 853, 857, 1373, 351, 1888, 1889, 354, 355, 869,
    358, 871, 1391, 884, 373, 1398, 1914, 1917, 894, 1920,
    1924, 1925, 1418, 909, 1422, 910, 1425, 1937, 1427, 1428,
    1430, 1436, 1954, 933, 1958, 429, 1970, 436, 1973, 1976,
    1977, 440, 1985, 1476, 966, 968, 970, 460, 1485, 973, 1489,
    468, 1495, 2008, 993, 2021, 493, 1014, 1528, 505, 1532,
    1533, 2047
])

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
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        try:
            try:
                with np.load(checkpoint_centers, allow_pickle=True) as data:
                    try:
                        original_centers = data['cluster_centers']
                    except KeyError as e:
                        print(f"Error: Missing 'cluster_centers' key in NPZ file: {e}")
                        raise
            except FileNotFoundError as e:
                print(f"Error: checkpoint file not found: {e}")
                raise

            n_clusters = original_centers.shape[0]
            valid_indices = [i for i in range(n_clusters) if i not in BLOCKLIST_CLUSTERS]
            print(f"Original # centers: {n_clusters}")
            print(f"Blocklist size: {len(BLOCKLIST_CLUSTERS)}")
            print(f"Valid centers after filtering: {len(valid_indices)}")
            cluster_centers = original_centers[valid_indices]
            self.cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32).to(self.device)
            self.kept_cluster_indices = valid_indices
            print("Style cluster centers loaded.")
        except Exception as e:
            print(f"Failed to load style cluster centers: {e}")

    @torch.no_grad()
    def get_cluster(self, features):
        """指定されたfeaturesからスタイルクラスタID（文字列）を返す"""
        distances = torch.norm(self.cluster_centers - features, dim=1)
        min_index = torch.argmin(distances).item()
        distance = distances[min_index]
        original_cluster_id = self.kept_cluster_indices[min_index]
        return original_cluster_id, distance

    @torch.no_grad()
    def get_cluster_batch(self, features: torch.Tensor | np.ndarray):
        """
        バッチ版:
        features: (N, D)
        return: (cluster_id_list, distance_ndarray)
        """
        if not torch.is_tensor(features):
            features = torch.from_numpy(features)

        features = features.to(self.device)  # (N,D)
        # (N, num_centers) の距離行列
        # cdist は多少重いけど、python for で回すより遥かに速い
        distances = torch.cdist(features, self.cluster_centers)  # (N, C)

        min_dists, min_idx = distances.min(dim=1)  # (N,)

        # 元のクラスタIDに戻す
        cluster_ids = [
            self.kept_cluster_indices[int(i)]
            for i in min_idx.cpu().tolist()
        ]
        return cluster_ids, min_dists.cpu().numpy()


# =========================
# WD Tagger (SwinV2)
# =========================

class Tagger:
    """SmilingWolf/wd-swinv2 ベースのタグ付けクラス."""

    def __init__(self, hf_token: str | None):
        self.hf_token = hf_token
        self.model_target_size: int | None = None
        self.last_loaded_repo: str | None = None
        self.model: rt.InferenceSession | None = None
        self.tag_names: list[str] | None = None
        self.rating_indexes: list[int] | None = None
        self.general_indexes: list[int] | None = None
        self.character_indexes: list[int] | None = None

    def _download_model(self, model_repo: str) -> tuple[str, str]:
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
        )
        return csv_path, model_path

    def load_model(self, model_repo: str) -> None:
        """必要であればモデルをロード（同じ repo なら再利用）"""
        if model_repo == self.last_loaded_repo and self.model is not None:
            return

        csv_path, model_path = self._download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        providers = ["CUDAExecutionProvider"]
        model = rt.InferenceSession(model_path, providers=providers)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """wd-tagger 用の前処理."""
        if self.model_target_size is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        target_size = self.model_target_size

        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize and pad the image
        image_shape = image.size  # (w, h)
        max_dim = max(image_shape)
        scale_factor = target_size / max_dim

        new_size = tuple(int(dim * scale_factor) for dim in image_shape)
        resized_image = image.resize(new_size, Image.LANCZOS)

        # Create a white background
        padded_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))

        # Paste the resized image onto the white background
        paste_pos = (
            (target_size - new_size[0]) // 2,
            (target_size - new_size[1]) // 2,
        )
        padded_image.paste(resized_image, paste_pos)

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)


    def _prepare_images_batch(self, images: list[Image.Image]) -> np.ndarray:
        """
        バッチ版: 画像リストを受け取って (N, H, W, 3) にまとめる
        """
        if self.model_target_size is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prepared = [self._prepare_image(img)[0] for img in images]

        return np.stack(prepared, axis=0)  # (N, H, W, 3)


    def predict(
        self,
        image: Image.Image,
        model_repo: str,
        general_thresh: float = 0.35,
        general_mcut_enabled: bool = False,
        character_thresh: float = 0.85,
        character_mcut_enabled: bool = False,
    ):
        """画像 1 枚分のタグ推論."""
        self.load_model(model_repo)

        try:
            image_arr = self._prepare_image(image)
        except OSError as e:
            print(f"Error: {e}")
            return None, None, None, None

        assert self.model is not None
        assert self.tag_names is not None
        assert self.rating_indexes is not None
        assert self.general_indexes is not None
        assert self.character_indexes is not None

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image_arr})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Characters
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = ", ".join(sorted_general_strings).replace(
            "(", "\\("
        ).replace(")", "\\)")

        return sorted_general_strings, rating, character_res, general_res


    def predict_batch(
        self,
        images: list[Image.Image],
        model_repo: str,
        general_thresh: float = 0.35,
        general_mcut_enabled: bool = False,
        character_thresh: float = 0.85,
        character_mcut_enabled: bool = False,
    ) -> list[tuple[str, dict, dict, dict]]:
        """
        バッチ版:
        return は predict と同じタプルを N 個並べたリスト
        """
        self.load_model(model_repo)
        image_arr = self._prepare_images_batch(images)  # (N,H,W,3)

        assert self.model is not None
        assert self.tag_names is not None
        assert self.rating_indexes is not None
        assert self.general_indexes is not None
        assert self.character_indexes is not None

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image_arr})[0]  # (N, num_tags)

        results = []
        for p in preds:
            labels = list(zip(self.tag_names, p.astype(float)))

            ratings_names = [labels[i] for i in self.rating_indexes]
            rating = dict(ratings_names)

            general_names = [labels[i] for i in self.general_indexes]

            _general_thresh = general_thresh
            if general_mcut_enabled:
                general_probs = np.array([x[1] for x in general_names])
                _general_thresh = mcut_threshold(general_probs)

            general_res = [x for x in general_names if x[1] > _general_thresh]
            general_res = dict(general_res)

            character_names = [labels[i] for i in self.character_indexes]
            _character_thresh = character_thresh
            if character_mcut_enabled:
                character_probs = np.array([x[1] for x in character_names])
                _character_thresh = max(0.15, mcut_threshold(character_probs))

            character_res = [x for x in character_names if x[1] > _character_thresh]
            character_res = dict(character_res)

            sorted_general_strings = sorted(
                general_res.items(), key=lambda x: x[1], reverse=True
            )
            sorted_general_strings = [x[0] for x in sorted_general_strings]
            sorted_general_strings = ", ".join(sorted_general_strings).replace(
                "(", "\\("
            ).replace(")", "\\)")

            results.append(
                (sorted_general_strings, rating, character_res, general_res)
            )

        return results

# =========================
# Z3D Tagger
# =========================

class Z3DTagger:
    """toynya/Z3D-E621-Convnext ベースのタグ付けクラス."""

    def __init__(
        self,
        hf_token: str | None,
        model_repo: str = "toynya/Z3D-E621-Convnext",
        threshold: float = 0.5,
    ):
        self.hf_token = hf_token
        self.model_repo = model_repo
        self.threshold = threshold

        self.tags: list[tuple[str, str]] = []
        csv_path, model_path = self._download_model(model_repo)

        with open(csv_path, mode="r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.tags.append(
                    (row["name"].strip().replace("_", " "), row["category"])
                )

        self.session = rt.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"],
        )

    def _download_model(self, model_repo: str) -> tuple[str, str]:
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            "tags-selected.csv",
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            "model.onnx",
        )
        return csv_path, model_path

    @staticmethod
    def _prepare_image(image: Image.Image, target_size: int) -> np.ndarray:
        """Z3D 用の前処理."""
        # Pad image to square
        image_shape = image.size  # (w, h)
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)


    @staticmethod
    def _prepare_images_batch(images: list[Image.Image], target_size: int) -> np.ndarray:
        arrs = []
        for image in images:
            image_shape = image.size  # (w, h)
            max_dim = max(image_shape)
            pad_left = (max_dim - image_shape[0]) // 2
            pad_top = (max_dim - image_shape[1]) // 2

            padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded_image.paste(image, (pad_left, pad_top))

            if max_dim != target_size:
                padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

            image_array = np.asarray(padded_image, dtype=np.float32)
            image_array = image_array[:, :, ::-1]  # RGB→BGR
            arrs.append(image_array)

        return np.stack(arrs, axis=0)  # (N, H, W, 3)


    def predict(self, image: Image.Image):
        """画像 1 枚分の Z3D 推論."""
        image_array = self._prepare_image(image, 448)
        input_name = "input_1:0"
        output_name = "predictions_sigmoid"

        result = self.session.run([output_name], {input_name: image_array})
        result = result[0][0]

        all_tags = {
            self.tags[i][0]: float(result[i])
            for i in range(len(result))
            if float(result[i]) > self.threshold
            and self.tags[i][1] in ["0", "3", "5"]
        }

        character_tags = {
            self.tags[i][0]: float(result[i])
            for i in range(len(result))
            if float(result[i]) > self.threshold
            and self.tags[i][1] == "4"
        }
        return all_tags, character_tags


    def predict_batch(self, images: list[Image.Image]):
        """
        バッチ版:
        - 画像はまとめて前処理する
        - ただし ONNX モデルのバッチ次元が 1 固定なので、
          実際の推論は 1 枚ずつループで session.run する
        return: list[(all_tags_dict, character_tags_dict)]
        """
        if not images:
            return []

        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape  # 例: [1, 448, 448, 3] など
        # バッチ次元（0番目）が 1 になっているはず
        expected_batch = input_shape[0]

        batch_results = []

        # 1枚ずつ処理（擬似バッチ）
        for img in images:
            # 単体前処理と同じ処理を流用
            arr = self._prepare_image(img, 448)  # shape: (1, H, W, 3)

            # 念のため shape チェック
            if isinstance(expected_batch, int) and expected_batch != 1:
                # もしここに来るならモデル側の仕様が想定外
                raise RuntimeError(
                    f"Z3D model batch dimension is {expected_batch}, expected 1."
                )

            output_name = "predictions_sigmoid"
            res = self.session.run([output_name], {input_name: arr})[0][0]  # (num_tags,)

            all_tags = {
                self.tags[i][0]: float(res[i])
                for i in range(len(res))
                if float(res[i]) > self.threshold and self.tags[i][1] in ["0", "3", "5"]
            }
            character_tags = {
                self.tags[i][0]: float(res[i])
                for i in range(len(res))
                if float(res[i]) > self.threshold and self.tags[i][1] == "4"
            }
            batch_results.append((all_tags, character_tags))

        return batch_results


# =========================
# 高レベルのサービスクラス
# =========================

class ImageTaggingService:
    """
    ・2 つの Tagger（wd + Z3D）をまとめて扱う
    ・1 枚の画像 or フォルダ全体を処理する
    """

    def __init__(
        self,
        hf_token: str | None = None,
        swin_repo: str = "SmilingWolf/wd-swinv2-tagger-v3",
        z3d_repo: str = "toynya/Z3D-E621-Convnext",
        general_thresh: float = 0.35,
        character_thresh: float = 0.85,
        z3d_threshold: float = 0.5,
    ):
        self.swin_repo = swin_repo
        self.general_thresh = general_thresh
        self.character_thresh = character_thresh

        self.wd_tagger = Tagger(hf_token)
        self.z3d_tagger = Z3DTagger(
            hf_token,
            model_repo=z3d_repo,
            threshold=z3d_threshold,
        )

    def tag_image(self, image: Image.Image) -> dict:
        """PIL 画像 1 枚をタグ付けして dict を返す."""

        if image.mode != "RGB":
            image = image.convert("RGB")

        # wd-swinv2
        general_tags, rating, character_tags, all_general_tags = \
            self.wd_tagger.predict(
                image,
                model_repo=self.swin_repo,
                general_thresh=self.general_thresh,
                general_mcut_enabled=False,
                character_thresh=self.character_thresh,
                character_mcut_enabled=False,
            )

        # Z3D
        all_general_tags_z3d, character_tags_z3d = self.z3d_tagger.predict(image)

        # 結合
        full_character_tags = set(character_tags.keys()) | set(character_tags_z3d.keys())
        full_general_tags = set(all_general_tags.keys()) | set(all_general_tags_z3d.keys())

        tags: list[str] = []
        tags.extend(list(full_general_tags))

        for tag in full_character_tags:
            tags.append(f"character:{tag}")

        top_rating = max(rating.items(), key=lambda x: x[1])

        return {"rating": top_rating[0], "tags": tags}


    def tag_image_pil(self, image: Image.Image) -> dict:
        """PIL 画像を直接受ける単体版（ヘルパー）"""
        return self.tag_image(image)


    def tag_images_batch(self, images: list[Image.Image]) -> list[dict]:
        """
        バッチ版: PIL 画像のリストを受け取り、各画像について tag_image と同じ dict を返す
        """

        processed_images: list[Image.Image | None] = []
        for im in images:
            if im is None:
                processed_images.append(None)
                continue
            processed_images.append(im.convert("RGB"))

        valid_images: list[Image.Image] = []
        valid_indices: list[int] = []
        for i, im in enumerate(processed_images):
            if im is None:
                continue
            valid_images.append(im)
            valid_indices.append(i)

        if not valid_images:
            return [{"rating": "", "tags": []} for _ in images]

        wd_results = self.wd_tagger.predict_batch(
            valid_images,
            model_repo=self.swin_repo,
            general_thresh=self.general_thresh,
            general_mcut_enabled=False,
            character_thresh=self.character_thresh,
            character_mcut_enabled=False,
        )
        z3d_results = self.z3d_tagger.predict_batch(valid_images)

        out: list[dict | None] = [None for _ in images]
        for local_idx, original_idx in enumerate(valid_indices):
            general_tags, rating, character_tags, all_general_tags = wd_results[local_idx]
            all_general_tags_z3d, character_tags_z3d = z3d_results[local_idx]

            full_character_tags = set(character_tags.keys()) | set(character_tags_z3d.keys())
            full_general_tags = set(all_general_tags.keys()) | set(all_general_tags_z3d.keys())

            tags: list[str] = list(full_general_tags)
            for tag in full_character_tags:
                tags.append(f"character:{tag}")

            top_rating = max(rating.items(), key=lambda x: x[1])
            out[original_idx] = {"rating": top_rating[0], "tags": tags}

        # None だったものを空の結果で埋める
        for i, res in enumerate(out):
            if res is None:
                out[i] = {"rating": "", "tags": []}

        return out


    def process_folder(
        self,
        image_folder: str,
        image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp"),
        force_tag_regen: bool = False,
    ):
        """
        フォルダ内の画像を一括タグ付けし、
        `xxx.tags.json` を出力する。
        """
        image_folder = str(image_folder)

        for filename in os.listdir(image_folder):
            if not filename.lower().endswith(image_extensions):
                continue

            image_path = os.path.join(image_folder, filename)
            name, _ = os.path.splitext(filename)
            tags_file = os.path.join(image_folder, f"{name}.tags.json")

            if force_tag_regen or not os.path.exists(tags_file):
                with Image.open(image_path) as image:
                    tags = self.tag_image(image.convert("RGB"))
                with open(tags_file, "w", encoding="utf-8") as f:
                    json.dump(tags, f, ensure_ascii=False, indent=2)
                print(f"Created tags file for: {filename}")
            else:
                print(f"Caption tags exists for: {filename}")


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

    parser.add_argument("--image_dir", help="dir", default="//192.168.1.46/ikutoDataset/dataset/gallery-dl")
    parser.add_argument("--meta_dir", help="dir", default="C:/Users/ikuto/projects/clip_meta")

    parser.add_argument(
        "--aesthetic_model_path", help="aesthetic_model_path", default="./model/aesthetic_ranking100.pth"
    )
    parser.add_argument(
        "--search_model_name", help="model_name", default="ViT-L-14"
    )
    parser.add_argument(
        "--search_model_pretrained", help="pretrained", default="openai"
    )
    parser.add_argument(
        "--search_model_out_dim", help="search_model_out_dim", type=int, default=768
    )

    parser.add_argument(
        "--aesthetic_checkpoint", help="aesthetic_checkpoint", default=AESTHETIC_REPO
    )

    parser.add_argument(
        "--style_checkpoint_model", help="style_checkpoint_model", default=STYLE_REPO
    )
    parser.add_argument(
        "--style_checkpoint_centers", help="style_checkpoint_centers", default=STYLE_REPO
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--num_workers", help="data loader workers", type=int, default=16)
    parser.add_argument("--nlist", help="centroid size", type=int, default=64)
    parser.add_argument("--M", help="M", type=int, default=768)
    parser.add_argument("--bits_per_code", help="bits_per_code", type=int, default=8)
    parser.add_argument(
        "--metas_faiss_index_file_name",
        help="metas_faiss_index_file_name",
        default="metafiles.index",
    )

    return parser.parse_args()


_PARALLEL = 64


def get_default_image_extensions() -> list[str]:
    """Return a list of image file extensions supported by Pillow.

    Using :func:`Image.registered_extensions` avoids maintaining a hard-coded
    extension list and keeps us aligned with the formats Pillow can open.
    """

    # Keep the ordering stable for deterministic behavior when printing or
    # iterating the extensions list.
    return sorted(Image.registered_extensions().keys())

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
            style_cluster text,
            rating text,
            image_tags text,
            time_stamp_ISO text
        )
        """
    )


def load_image_meta_from_db(
    con: sqlite3.Connection, id_list: list[str], loop_size: int
) -> Iterable[tuple[str, Any]]:

    if not id_list:
        return []

    cur = con.cursor()
    temp_table = f"valid_id_{uuid.uuid4().hex}"

    cur.execute(f"DROP TABLE IF EXISTS \"{temp_table}\"")
    cur.execute(
        f"CREATE TEMP TABLE \"{temp_table}\" (image_id TEXT PRIMARY KEY)"
    )

    for idx in tqdm.tqdm(range(0, len(id_list), loop_size)):
        sub_list: list[str] = id_list[idx: idx + loop_size]
        cur.executemany(
            f"INSERT INTO \"{temp_table}\" (image_id) VALUES (?)",
            ((image_id,) for image_id in sub_list),
        )

    cur.execute(
        f"""
        SELECT
            valid_id.image_id,
            image_meta.meta
        FROM
            \"{temp_table}\" AS valid_id
            LEFT JOIN image_meta ON valid_id.image_id = image_meta.image_id
        """
    )

    try:
        while True:
            rows = cur.fetchmany(loop_size)
            if not rows:
                break
            for row in rows:
                yield row
    finally:
        cur.execute(f"DROP TABLE IF EXISTS \"{temp_table}\"")


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
    id_to_path: dict[str, str] = {}
    path_to_id: dict[str, str] = {}

    for file in file_list:
        image_id = hashlib.sha256(str(file).encode()).hexdigest()
        id_to_path[image_id] = file
        path_to_id[file] = image_id

    return id_to_path, path_to_id


def filter_valid_image_meta(args, result, id_to_path):
    search_meta_list = []
    uncreated_image_paths = []
    for image_id, meta in tqdm.tqdm(result, total=len(id_to_path)):
        if meta is not None:
            meta = np.squeeze(meta)
            a = np.frombuffer(meta, dtype=np.float32)
            if a.shape[0] == args.search_model_out_dim:
                search_meta_list.append(a)
                continue

        uncreated_image_paths.append(id_to_path[image_id])
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
    con,
    cur,
    loader,
    uncreated_image_paths,
    path_to_id,
    aesthetic_model,
    pony_scorer,
    style_cluster,
    tagging_service,
):

    processed_count = 0
    processed_batches = 0
    commit_interval = max(1, 10000 // args.batch_size)

    for batch in tqdm.tqdm(loader, total=len(uncreated_image_paths) // args.batch_size + 1):
        if batch is None:
            continue

        batched_image_input, batched_image_index, batched_raw_images = batch  # (B, C, H, W), (B,), (B,)

        batched_image_input = batched_image_input.to(device, non_blocking=True)
        try:
            # visual と proj を取り出す
            visual = search_model.visual
            proj = getattr(visual, "proj", None)

            if proj is not None:
                # proj を一時的に外して「内部特徴」を取得
                visual.proj = None
                internal_features_tensor = visual(batched_image_input)  # (B, D_internal)
                visual.proj = proj

                # 内部特徴 → CLIP 埋め込み（768次元など）に射影
                image_features_tensor = internal_features_tensor @ proj        # (B, D_clip)
            else:
                # proj が無い場合はそのまま
                internal_features_tensor = visual(batched_image_input)
                image_features_tensor = internal_features_tensor


            denom = image_features_tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            image_features_tensor = (image_features_tensor / denom).contiguous()
            aesthetic_scores = torch.sigmoid(
                aesthetic_model(image_features_tensor)
            ).squeeze(-1).cpu().numpy()  # (B,)

            pony_scores = pony_scorer.score_batch(image_features_tensor)  # (B,)

            cluster_ids, _cluster_dists = style_cluster.get_cluster_batch(
                internal_features_tensor
            )  # cluster_ids: list[str], len=B

            # 3) タグ付けもバッチで
            tagging_results = tagging_service.tag_images_batch(
                list(batched_raw_images)
            )  # list[dict], len=B


            batched_new_search_meta = (
                image_features_tensor
                .to("cpu")
                .detach()
                .numpy()
                .copy()
                .astype(np.float32)
            )

            params = []

            for i, (new_search_meta, image_index, aesthetic_score, pony_aesthetic_score, cluster_id, tag_res) in enumerate(
                zip(
                    batched_new_search_meta,
                    batched_image_index,
                    aesthetic_scores,
                    pony_scores,
                    cluster_ids,
                    tagging_results,
                )
            ):
                if new_search_meta.shape[0] != args.search_model_out_dim:
                    print(f"{new_search_meta.shape[0]} != {args.search_model_out_dim}")
                    continue
                new_search_meta_bytes = new_search_meta.tobytes()
                rating = tag_res.get("rating", "")
                image_tags = ", ".join(tag_res.get("tags", []))
                claster = f"style_cluster_{cluster_id}" if cluster_id is not None else ""

                image_index_int = int(image_index)
                image_path = uncreated_image_paths[image_index_int]
                image_id = path_to_id.get(image_path)
                if image_id is None:
                    print(f"[extract_image_features] image_id not found for {image_path}")
                    continue
                time_stamp_ISO = datetime.datetime.now(datetime.timezone.utc).isoformat()

                params.append((
                    image_id,
                    image_path,
                    new_search_meta_bytes,
                    float(aesthetic_score),
                    float(pony_aesthetic_score),
                    claster,
                    rating,
                    image_tags,
                    time_stamp_ISO,
                    new_search_meta_bytes,
                    float(aesthetic_score),
                    float(pony_aesthetic_score),
                    claster,
                    rating,
                    image_tags,
                    time_stamp_ISO,
                ))

            if params:
                cur.executemany(
                    """insert into image_meta values(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(image_id) do update set
                        meta=?,
                        aesthetic_quality=?,
                        pony_aesthetic_quality=?,
                        style_cluster=?,
                        rating=?,
                        image_tags=?,
                        time_stamp_ISO=?""",
                    params,
                )

            processed_count += len(params)
            if params:
                processed_batches += 1
                if processed_batches % commit_interval == 0:
                    print(f"commit after batch {processed_batches} (total rows: {processed_count})")
                    con.commit()

        except Exception:
            traceback.print_exc()
            continue


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

    ext_list = get_default_image_extensions()

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

    aesthetic_checkpoint_arg = getattr(args, "aesthetic_checkpoint", None)
    style_checkpoint_model_arg = getattr(args, "style_checkpoint_model", None)
    style_checkpoint_centers_arg = getattr(args, "style_checkpoint_centers", None)

    aesthetic_checkpoint = ensure_asset_path(
        aesthetic_checkpoint_arg,
        AESTHETIC_CHECKPOINT_FILENAME,
        "aesthetic checkpoint",
    )
    style_checkpoint_model = ensure_asset_path(
        style_checkpoint_model_arg,
        STYLE_MODEL_FILENAME,
        "style classifier checkpoint",
    )
    style_checkpoint_centers = ensure_asset_path(
        style_checkpoint_centers_arg,
        STYLE_CENTERS_FILENAME,
        "style cluster centers",
    )

    aesthetic_model = get_aesthetic_model(args.aesthetic_model_path, clip_model="vit_l_14")
    aesthetic_model = aesthetic_model.to(device)
    aesthetic_model.eval()

    pony_scorer = PonyAestheticScorer(device=device, checkpoint=aesthetic_checkpoint)
    style_cluster = StyleCluster(
        device=device,
        checkpoint_model=style_checkpoint_model,
        checkpoint_centers=style_checkpoint_centers,
    )
    tagging_service = ImageTaggingService(hf_token=None)

    id_to_path, path_to_id = create_image_id_index(index_item_list)

    # データベースに問い合わせる
    result = load_image_meta_from_db(con, id_list=list(id_to_path.keys()), loop_size=10000)

    search_meta_list, uncreated_image_paths = filter_valid_image_meta(args, result, id_to_path)

    print_image_meta_stats(search_meta_list, uncreated_image_paths, len(id_to_path))

    del result, id_to_path, search_meta_list

    dataset = None
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
            target_paths = uncreated_image_paths[i * args.batch_size :]
            if not target_paths:
                break

            dataset = DirImageIterable(
                args.image_dir,
                target_paths,
                transform=eval_transform,
            )
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
                    con,
                    cur,
                    loader,
                    target_paths,
                    path_to_id,
                    aesthetic_model,
                    pony_scorer,
                    style_cluster,
                    tagging_service,
                )
                T = False
            except Exception:
                traceback.print_exc()
                i += 1
                continue

    con.commit()

    del uncreated_image_paths, search_model, dataset, loader
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
