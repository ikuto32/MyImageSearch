from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import datetime
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
import pathlib
import sqlite3
import traceback
import typing
from typing import Any, Iterable
from urllib.parse import urlparse
import csv
import uuid
import time
import threading

import faiss
import huggingface_hub
import numpy as np
import open_clip

from app.infrastructure.model_metadata import (
    SearchModelMetadata,
    write_model_metadata,
)
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate
import onnxruntime as rt
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 268_435_456


def normalize_qwen_api_base(api_base: str) -> str:
    """Normalize a vLLM OpenAI-compatible API base URL so it ends in /v1."""

    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("Qwen API base URL must not be empty.")
    normalized = api_base.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Invalid Qwen API base URL: {api_base!r}. Expected an http(s) URL such as http://127.0.0.1:8000/v1."
        )
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def pil_image_to_data_url(image: Image.Image) -> str:
    """Convert a PIL image to a PNG data URL for the vLLM OpenAI API."""

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

# =========================
# 共通ヘルパー
# =========================

def _collate_search_inputs(search_inputs: list[Any]):
    if search_inputs and all(isinstance(item, Image.Image) for item in search_inputs):
        return search_inputs
    return default_collate(search_inputs)


def _collate_optional(values: list[Any]):
    if all(value is None for value in values):
        return None
    return default_collate(values)


def safe_collate(batch):
    """Collate search, metadata, index, and tagger inputs safely."""

    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    search_inputs, metadata_inputs, indices, wd_inputs, z3d_inputs = zip(*batch)

    return (
        _collate_search_inputs(list(search_inputs)),
        _collate_optional(list(metadata_inputs)),
        default_collate(list(indices)),
        _collate_optional(list(wd_inputs)),
        _collate_optional(list(z3d_inputs)),
    )


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


def prepare_tag_input(image: Image.Image, target_size: int) -> np.ndarray:
    """
    Tagging 共通の前処理を行う純関数.

    RGB化 → 縦横比維持のリサイズ → 白背景へのパディング → BGR/float32 へ変換.
    バッチ次元は付与しない (H, W, 3) を返す。
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    max_dim = max(width, height)
    scale = target_size / max_dim
    new_size = (int(width * scale), int(height * scale))

    resized_image = image.resize(new_size, Image.LANCZOS)

    padded_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    paste_pos = (
        (target_size - new_size[0]) // 2,
        (target_size - new_size[1]) // 2,
    )
    padded_image.paste(resized_image, paste_pos)

    image_array = np.asarray(padded_image, dtype=np.float32)
    # Channel reversal introduces negative strides; copy to keep collate happy.
    image_array = image_array[:, :, ::-1].copy()  # RGB → BGR

    return image_array


class DirImageIterable(IterableDataset):
    def __init__(
        self,
        images_dir: str,
        img_list,
        transform=None,
        *,
        wd_target_size: int,
        metadata_transform=None,
    ):
        self.images_dir = images_dir
        self.img_list = list(img_list)
        self.transform = transform
        self.metadata_transform = metadata_transform
        self.wd_target_size = wd_target_size

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

                try:
                    wd_arr = prepare_tag_input(img, self.wd_target_size)
                    z3d_arr = prepare_tag_input(img, 448)
                except Exception as e:
                    print(f"[DirImageIterable] tag preprocessing failed: {path} in {worker_id} -> {e}")
                    continue

                search_img = img
                metadata_img = None
                if self.transform is not None:
                    try:
                        search_img = self.transform(img)
                    except Exception as e:
                        print(f"[DirImageIterable] search transform failed: {path} in {worker_id} -> {e}")
                        continue
                if self.metadata_transform is not None:
                    try:
                        metadata_img = self.metadata_transform(img)
                    except Exception as e:
                        print(f"[DirImageIterable] metadata transform failed: {path} in {worker_id} -> {e}")
                        continue
                try:
                    yield search_img, metadata_img, idx, wd_arr, z3d_arr
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
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.SiLU())
        self.dropout2 = nn.Dropout1d(0.3)
        self.fc3 = nn.Sequential(nn.Linear(512, 1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def score_batch(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, D) → (N,) の美的評価スコアを返す"""
        norm = features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        norm[norm == 0] = 1
        features = features / norm
        with torch.no_grad():
            scores = self.forward(features).squeeze(-1)
        return scores.clamp(0, 1)


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
            aesthetic_score = self.aesthetic_model(features).clamp(0, 1).item()
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

    def _prepare_style_features(self, features: torch.Tensor | np.ndarray, *, batched: bool):
        if not torch.is_tensor(features):
            features = torch.from_numpy(features)

        expected_ndim = 2 if batched else 1
        if features.ndim != expected_ndim:
            shape_label = "(N, D)" if batched else "(D,)"
            raise ValueError(
                f"Style features must have shape {shape_label}, "
                f"got {tuple(features.shape)}"
            )

        expected_dim = int(self.cluster_centers.shape[1])
        actual_dim = int(features.shape[1] if batched else features.shape[0])
        if actual_dim != expected_dim:
            raise ValueError(
                "Style feature dimension mismatch: "
                f"features={actual_dim}, "
                f"cluster_centers={expected_dim}. "
                "Use OpenCLIP pre-projection/internal features "
                "for style clustering."
            )

        return features.to(
            device=self.cluster_centers.device,
            dtype=self.cluster_centers.dtype,
            non_blocking=True,
        )

    @torch.no_grad()
    def get_cluster(self, features: torch.Tensor | np.ndarray):
        """指定された単一のstyle features (D,) からスタイルクラスタIDを返す。"""
        features = self._prepare_style_features(features, batched=False)
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
        features = self._prepare_style_features(features, batched=True)
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

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        so = rt.SessionOptions()
        so.log_severity_level = 2
        so.log_verbosity_level = 0
        # so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        model = rt.InferenceSession(model_path, providers=providers, sess_options=so)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model



    def predict_batch(
        self,
        image_arrs: np.ndarray,
        *,
        model_repo: str,
        general_thresh: float = 0.35,
        general_mcut_enabled: bool = False,
        character_thresh: float = 0.85,
        character_mcut_enabled: bool = False,
    ):
        """前処理済み入力を受け取ってバッチ推論する."""

        if image_arrs.size == 0:
            return []

        self.load_model(model_repo)

        assert self.model is not None
        assert self.tag_names is not None
        assert self.rating_indexes is not None
        assert self.general_indexes is not None
        assert self.character_indexes is not None

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image_arrs})[0]

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
                "(", r"\("
            ).replace(")", r"\)")

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
        so = rt.SessionOptions()
        so.log_severity_level = 2
        so.log_verbosity_level = 0
        # so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = rt.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            sess_options=so,
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


    def predict_batch(self, image_arrs: np.ndarray):
        """前処理済み配列からタグを推論."""

        if image_arrs.size == 0:
            return []

        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        expected_batch = input_shape[0]
        output_name = "predictions_sigmoid"

        batch_results = []
        image_arrs = image_arrs.astype(np.float32, copy=False)

        def _to_tags(res: np.ndarray):
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
            return all_tags, character_tags

        if isinstance(expected_batch, int) and expected_batch == 1:
            for arr in image_arrs:
                arr_in = np.expand_dims(arr, axis=0)
                res = self.session.run([output_name], {input_name: arr_in})[0][0]
                batch_results.append(_to_tags(res))
        else:
            res_batch = self.session.run([output_name], {input_name: image_arrs})[0]
            for res in res_batch:
                batch_results.append(_to_tags(res))

        return batch_results

# =========================
# 高レベルのサービスクラス
# =========================


class ImageTaggingService:
    """
    ・2 つの Tagger（wd + Z3D）をまとめて扱う
    ・前処理済みバッチ入力を受け取ってタグ付けする
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

    def ensure_wd_model_loaded(self) -> int:
        self.wd_tagger.load_model(self.swin_repo)
        if self.wd_tagger.model_target_size is None:
            raise RuntimeError("WD tagger model size is not available")
        return self.wd_tagger.model_target_size

    def tag_batch(self, wd_batch: np.ndarray, z3d_batch: np.ndarray) -> list[dict]:
        if wd_batch.size == 0:
            return []

        wd_results = self.wd_tagger.predict_batch(
            wd_batch,
            model_repo=self.swin_repo,
            general_thresh=self.general_thresh,
            general_mcut_enabled=False,
            character_thresh=self.character_thresh,
            character_mcut_enabled=False,
        )
        z3d_results = self.z3d_tagger.predict_batch(z3d_batch)

        out: list[dict] = []
        for wd_res, z3d_res in zip(wd_results, z3d_results):
            sorted_general_strings, rating, character_tags, all_general_tags = wd_res
            all_general_tags_z3d, character_tags_z3d = z3d_res

            full_character_tags = set(character_tags.keys()) | set(
                character_tags_z3d.keys()
            )
            full_general_tags = set(all_general_tags.keys()) | set(
                all_general_tags_z3d.keys()
            )

            tags: list[str] = list(full_general_tags)
            for tag in full_character_tags:
                tags.append(f"character:{tag}")

            top_rating = max(rating.items(), key=lambda x: x[1]) if rating else ("", 0)
            out.append({"rating": top_rating[0], "tags": tags})

        return out


def get_aesthetic_model(path_to_model, clip_model="vit_l_14"):
    """load the aethetic model"""
    if clip_model == "vit_l_14":
        m = Aesthetic_model()
    else:
        raise ValueError()
    raw = torch.load(path_to_model, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    m.load_state_dict(state_dict, strict=True)
    m.eval()
    return m


def positive_int(value: str) -> int:
    """Argparse type that accepts only positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    # parser.add_argument("--image_dir", help="dir", default="//192.168.1.46/ikutoDataset/dataset/gallery-dl")
    # parser.add_argument("--meta_dir", help="dir", default="C:/Users/ikuto/projects/clip_meta")
    parser.add_argument("--image_dir", help="dir", default="./images")
    parser.add_argument("--meta_dir", help="dir", default="./clip_meta")

    parser.add_argument(
        "--aesthetic_model_path", help="aesthetic_model_path", default="./model/aesthetic_rankingv2.pth"
    )
    parser.add_argument(
        "--search_backend",
        choices=("open_clip", "qwen_vl"),
        default="qwen_vl",
        help="Search embedding backend to use.",
    )
    parser.add_argument(
        "--search_model_name",
        help="OpenCLIP model_name (used only with --search_backend open_clip)",
        default="ViT-L-14",
    )
    parser.add_argument(
        "--search_model_pretrained",
        help="OpenCLIP pretrained tag (used only with --search_backend open_clip)",
        default="openai",
    )
    parser.add_argument(
        "--search_model_id",
        help="Hugging Face model ID for non-OpenCLIP backends.",
        default="Qwen/Qwen3-VL-Embedding-2B",
    )
    parser.add_argument(
        "--search_model_out_dim",
        help=(
            "Optional output embedding dimension. If omitted, the dimension is "
            "inferred from the first generated/stored embedding. Qwen MRL models "
            "are truncated to this size when specified."
        ),
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--qwen-max-pixels",
        dest="qwen_max_pixels",
        help=(
            "Deprecated compatibility option for Qwen VL. Ignored by the API backend; "
            "configure max_pixels with vLLM --mm-processor-kwargs instead."
        ),
        type=positive_int,
        default=None,
    )
    parser.add_argument("--qwen-api-base", default=os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--qwen-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--qwen-api-timeout", type=positive_float, default=300.0)
    parser.add_argument("--qwen-api-max-retries", type=nonnegative_int, default=2)
    parser.add_argument("--qwen-api-concurrency", type=positive_int, default=4)
    parser.add_argument("--qwen-instruction", default="Represent the user's input.")
    parser.add_argument("--qwen-api-skip-server-check", action="store_true")

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
    parser.add_argument("--num_workers", help="data loader workers", type=int, default=8)
    parser.add_argument("--nlist", help="centroid size", type=int, default=64)
    parser.add_argument(
        "--pq-m",
        "--M",
        dest="pq_m",
        help="FAISS IVFPQ PQ subvector count (not embedding dimension).",
        type=int,
        default=64,
    )
    parser.add_argument("--bits_per_code", help="bits_per_code", type=int, default=8)
    parser.add_argument(
        "--metas_faiss_index_file_name",
        help="metas_faiss_index_file_name",
        default="metafiles.index",
    )
    parser.add_argument(
        "--use-existing-tags",
        action="store_true",
        help="Skip tagging and read precomputed *.tags.json files when available",
    )
    parser.add_argument(
        "--scan-timeout-sec",
        type=float,
        default=120,
        help=(
            "Directory scan stall timeout in seconds. A scan is skipped only when "
            "no progress is observed for this duration."
        ),
    )
    parser.add_argument(
        "--delete-orphan-db-records",
        action="store_true",
        help=(
            "Delete records that exist only in DB (not in the current image directory). "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--recalc-aesthetic",
        action="store_true",
        help=(
            "DBに保存済みの CLIP ベクトル (meta) を再利用して aesthetic_quality / "
            "pony_aesthetic_quality のみを高速再計算し、DB を更新して FAISSインデックスを再構築する。"
            "画像ファイルのスキャン・CLIP推論・タグ付けはすべてスキップされる。"
        ),
    )
    parser.add_argument(
        "--recalc-aesthetic-batch-size",
        type=int,
        default=16384,
        help="recalc-aesthetic モード時の DB フェッチ・更新バッチサイズ (default: 16384)",
    )
    parser.add_argument(
        "--disable-clip-metadata",
        action="store_true",
        help=(
            "検索バックエンドと別に 768 次元 OpenCLIP メタデータ特徴量を生成せず、"
            "aesthetic/style メタデータを NULL/空で保存する。主に Qwen インデックス用。"
        ),
    )

    return parser.parse_args()


_PARALLEL = 64
_SCAN_TIMEOUT_SEC = 3600
_SCAN_STALL_LOG_INTERVAL_SEC = 300


@dataclass
class _ScanProgress:
    started_at: float
    last_progress_at: float


def get_default_image_extensions() -> list[str]:
    """Return a list of image file extensions supported by Pillow.

    Using :func:`Image.registered_extensions` avoids maintaining a hard-coded
    extension list and keeps us aligned with the formats Pillow can open.
    """

    # Keep the ordering stable for deterministic behavior when printing or
    # iterating the extensions list.
    return sorted(Image.registered_extensions().keys())


def get_image_list_from_dir(
    dir_path: str | os.PathLike,
    exts: typing.Sequence[str],
    *,
    scan_timeout_sec: float = _SCAN_TIMEOUT_SEC,
) -> list[str]:
    """
    dir_path 以下を走査して、指定拡張子のファイルリスト（dir_path からの相対パス）を返す。
    SMB のレイテンシを考慮して並列数を抑えつつ、ディレクトリ単位でまとめてスレッドプールに流す。
    """
    return asyncio.run(
        _collect_images(pathlib.Path(dir_path), exts, scan_timeout_sec=scan_timeout_sec)
    )


async def _collect_images(
    root: pathlib.Path,
    exts: typing.Sequence[str],
    *,
    scan_timeout_sec: float = _SCAN_TIMEOUT_SEC,
) -> list[str]:
    loop = asyncio.get_running_loop()

    # .xyz と xyz の両方を許容
    ext_set = {("."+e if not e.startswith(".") else e).lower() for e in exts}

    files: list[str] = []
    bar = tqdm.tqdm(unit="file", dynamic_ncols=True)

    in_progress_scans: dict[pathlib.Path, _ScanProgress] = {}
    in_progress_lock = asyncio.Lock()
    thread_progress_lock = threading.Lock()

    # 同時実行制限
    sem = asyncio.Semaphore(_PARALLEL)

    def _mark_progress(path: pathlib.Path) -> None:
        now = time.monotonic()
        with thread_progress_lock:
            progress = in_progress_scans.get(path)
            if progress is not None:
                progress.last_progress_at = now

    def _get_progress_snapshot(path: pathlib.Path) -> _ScanProgress | None:
        with thread_progress_lock:
            progress = in_progress_scans.get(path)
            if progress is None:
                return None
            return _ScanProgress(
                started_at=progress.started_at,
                last_progress_at=progress.last_progress_at,
            )

    def _scan_dir_sync(path: pathlib.Path) -> tuple[list[pathlib.Path], list[str]]:
        """同期関数。path 配下を走査し、進捗更新しながら結果を返す。"""
        dirs, hits = [], []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    _mark_progress(path)
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

    async def _scan_with_timeout(path: pathlib.Path) -> tuple[list[pathlib.Path], list[str]]:
        start = time.monotonic()
        async with in_progress_lock:
            in_progress_scans[path] = _ScanProgress(started_at=start, last_progress_at=start)

        future = loop.run_in_executor(pool, _scan_dir_sync, path)
        try:
            while True:
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=scan_timeout_sec,
                    )
                except TimeoutError:
                    snapshot = _get_progress_snapshot(path)
                    if snapshot is None:
                        return await future

                    now = time.monotonic()
                    stall_sec = now - snapshot.last_progress_at
                    if stall_sec >= scan_timeout_sec:
                        print(
                            f"[scan timeout] path={path} "
                            f"last_progress_sec_ago={stall_sec:.1f} "
                            "scan stalled and will be skipped"
                        )
                        return [], []
        finally:
            async with in_progress_lock:
                in_progress_scans.pop(path, None)

    async def _stall_monitor() -> None:
        try:
            while True:
                await asyncio.sleep(_SCAN_STALL_LOG_INTERVAL_SEC)
                now = time.monotonic()
                async with in_progress_lock:
                    stalled = [
                        (
                            path,
                            now - progress.started_at,
                            now - progress.last_progress_at,
                        )
                        for path, progress in in_progress_scans.items()
                        if now - progress.started_at >= _SCAN_STALL_LOG_INTERVAL_SEC
                    ]

                if stalled:
                    stalled.sort(key=lambda x: x[2], reverse=True)
                    print("[scan monitor] slow directories currently being scanned:")
                    for path, elapsed, stall in stalled[:10]:
                        print(
                            f"  - {path} (elapsed={elapsed:.1f} sec, stalled={stall:.1f} sec)"
                        )
        except asyncio.CancelledError:
            return

    async def _walk(path: pathlib.Path):
        async with sem:
            # thread pool で同期スキャン
            dirs, hits = await _scan_with_timeout(path)
            files.extend(hits)
            bar.update(len(hits))

        # サブディレクトリを並列にたどる
        await asyncio.gather(*(_walk(d) for d in dirs))

    # プールは with で自動クローズ
    with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL) as pool:
        monitor_task = asyncio.create_task(_stall_monitor())
        try:
            await _walk(root)
        finally:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task

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
            time_stamp_ISO text,
            search_embedding_model text,
            metadata_embedding_model text
        )
        """
    )
    cur.execute("ALTER TABLE image_meta ADD COLUMN search_embedding_model text") if not _column_exists(cur, "image_meta", "search_embedding_model") else None
    cur.execute("ALTER TABLE image_meta ADD COLUMN metadata_embedding_model text") if not _column_exists(cur, "image_meta", "metadata_embedding_model") else None


def _column_exists(cur: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    return any(row[1] == column_name for row in cur.execute(f"PRAGMA table_info({table_name})"))


def load_image_meta_from_db(
    con: sqlite3.Connection, id_list: list[str], loop_size: int
) -> Iterable[tuple[str, Any]]:

    if not id_list:
        return []

    cur = con.cursor()
    temp_table = f"valid_id_{uuid.uuid4().hex}"

    try:
        cur.execute(f"DROP TABLE IF EXISTS \"{temp_table}\"")
        cur.execute(
            f"CREATE TEMP TABLE \"{temp_table}\" (image_id TEXT PRIMARY KEY)"
        )

        cur.execute("BEGIN")
        try:
            for idx in tqdm.tqdm(range(0, len(id_list), loop_size)):
                sub_list: list[str] = id_list[idx: idx + loop_size]
                cur.executemany(
                    f"INSERT INTO \"{temp_table}\" (image_id) VALUES (?)",
                    ((image_id,) for image_id in sub_list),
                )
        except Exception:
            con.rollback()
            raise
        else:
            con.commit()

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

        while True:
            rows = cur.fetchmany(loop_size)
            if not rows:
                break
            for row in rows:
                yield row
    finally:
        cur.execute(f"DROP TABLE IF EXISTS \"{temp_table}\"")


def createIndex(n_centroids, pq_m, bits_per_code, dim) -> faiss.IndexIVFPQ:

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, n_centroids, pq_m, bits_per_code, faiss.METRIC_INNER_PRODUCT
    )

    return index


class SearchEmbeddingBackend:
    """検索用の画像・テキスト埋め込みバックエンド共通インターフェース。"""

    @property
    def preprocess(self):
        raise NotImplementedError

    @torch.no_grad()
    def encode_image_with_internal(self, image_inputs):
        raise NotImplementedError

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError


class OpenClipEmbeddingBackend(SearchEmbeddingBackend):
    """既存の OpenCLIP ベース検索エンコーダ。"""

    def __init__(self, model_name: str, pretrained: str, device: str):
        self.device = device
        self.model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            jit=False,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def preprocess(self):
        return self._preprocess

    @torch.no_grad()
    def encode_image_with_internal(self, pixel_values: torch.Tensor):
        visual = self.model.visual
        x = visual._embeds(pixel_values)
        x = visual.transformer(x)
        pooled, _tokens = visual._pool(x)

        internal_features = pooled
        proj = getattr(visual, "proj", None)
        projected_features = pooled @ proj if proj is not None else pooled
        return internal_features, projected_features

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(tokens)


class QwenVlImagePreprocess:
    """Picklable image transform for Qwen VL search inputs.

    Qwen preprocessing is intentionally deferred to the main-process batch
    path because the processor owns non-trivial model state and may not be
    picklable under multiprocessing spawn. DataLoader workers only return
    lightweight PIL images.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.copy()


class QwenVlEmbeddingBackend(SearchEmbeddingBackend):
    """Qwen3-VL-Embedding backend using vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        model_id: str,
        output_dim: int | None = None,
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        max_retries: int = 2,
        max_concurrency: int = 4,
        instruction: str = "Represent the user's input.",
        skip_server_check: bool = False,
    ):
        if timeout <= 0:
            raise ValueError("qwen_api_timeout must be a positive number.")
        if max_retries < 0:
            raise ValueError("qwen_api_max_retries must be 0 or greater.")
        if max_concurrency < 1:
            raise ValueError("qwen_api_concurrency must be 1 or greater.")
        if not instruction:
            raise ValueError("qwen_instruction must not be empty.")

        self.model_id = model_id
        self.output_dim = output_dim
        self.api_base = normalize_qwen_api_base(api_base)
        self.api_key = api_key or "EMPTY"
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.max_concurrency = int(max_concurrency)
        self.instruction = instruction
        self._thread_local = threading.local()

        self.print_qwen_api_config()
        if not skip_server_check:
            self._check_server()

    def print_qwen_api_config(self) -> None:
        print("Qwen backend: vLLM OpenAI-compatible API")
        print(f"API base: {self.api_base}")
        print(f"served model: {self.model_id}")
        print(f"output dimension: {self.output_dim}")
        print(f"timeout: {self.timeout}")
        print(f"max retries: {self.max_retries}")
        print(f"API concurrency: {self.max_concurrency}")
        print(f"instruction: {self.instruction}")
        print(f"API key configured: {bool(self.api_key)}")

    @property
    def preprocess(self):
        return QwenVlImagePreprocess()

    def _get_client(self) -> OpenAI:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            self._thread_local.client = client
        return client

    def _check_server(self) -> None:
        try:
            models = self._get_client().models.list()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise self._runtime_api_error(
                "Could not connect to the Qwen vLLM server. Ensure the vLLM OpenAI-compatible server is running",
                exc,
            ) from exc
        model_ids = [getattr(model, "id", None) for model in getattr(models, "data", [])]
        model_ids = [model_id for model_id in model_ids if model_id]
        if self.model_id not in model_ids:
            available = ", ".join(model_ids) if model_ids else "<none>"
            raise RuntimeError(
                f"Qwen model '{self.model_id}' was not returned by {self.api_base}/models. "
                f"Available models: {available}. If vLLM was started with --served-model-name, "
                "set --search_model_id to that served name."
            )

    def _runtime_api_error(self, message: str, exc: Exception, *, index: int | None = None) -> RuntimeError:
        parts = [f"{message}; API base: {self.api_base}"]
        if index is not None:
            parts.append(f"batch index: {index}")
        if isinstance(exc, APIStatusError):
            parts.append(f"HTTP status: {exc.status_code}")
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    parts.append(f"response body: {response.text}")
                except Exception:
                    pass
        else:
            parts.append(f"error: {exc}")
        return RuntimeError("; ".join(parts))

    def _image_messages(self, image_data_url: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": ""},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

    def _text_messages(self, text: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

    def _embedding_body(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "messages": messages,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }

    def _post_embedding(self, messages: list[dict[str, Any]], *, index: int | None = None) -> np.ndarray:
        try:
            response = self._get_client().post(
                "/embeddings",
                cast_to=CreateEmbeddingResponse,
                body=self._embedding_body(messages),
            )
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            raise self._runtime_api_error("Qwen embedding API request failed", exc, index=index) from exc
        return self._validate_embedding_response(response, index=index)

    def _validate_embedding_response(self, response: Any, *, index: int | None = None) -> np.ndarray:
        prefix = f"Qwen embedding response for batch index {index}" if index is not None else "Qwen embedding response"
        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError(f"{prefix} did not contain any data.")
        embedding = getattr(data[0], "embedding", None)
        if embedding is None:
            raise RuntimeError(f"{prefix} did not contain data[0].embedding.")
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1:
            raise RuntimeError(f"{prefix} embedding must be a 1-dimensional numeric array; got shape {array.shape}.")
        if array.size == 0:
            raise RuntimeError(f"{prefix} embedding must not be empty.")
        if not np.isfinite(array).all():
            raise RuntimeError(f"{prefix} embedding contains NaN or Inf values.")
        if self.output_dim is not None:
            if array.shape[0] < self.output_dim:
                raise RuntimeError(
                    f"{prefix} embedding dimension {array.shape[0]} is smaller than requested output_dim {self.output_dim}."
                )
            array = array[: self.output_dim]
        return array.astype(np.float32, copy=False)

    def _stack_embeddings(self, embeddings: Sequence[np.ndarray]) -> torch.Tensor:
        stacked_embeddings = np.stack(embeddings).astype(np.float32, copy=False)
        return torch.from_numpy(stacked_embeddings).to(dtype=torch.float32)

    def _encode_messages_batch(self, messages: Sequence[list[dict[str, Any]]]) -> torch.Tensor:
        if not messages:
            dim = self.output_dim or 0
            return torch.empty((0, dim), dtype=torch.float32)
        max_workers = min(self.max_concurrency, len(messages))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(lambda item: self._post_embedding(item[1], index=item[0]), enumerate(messages)))
        return self._stack_embeddings(embeddings)

    def _post_image_embedding(self, item: tuple[int, Image.Image]) -> np.ndarray:
        index, image = item
        image_data_url = pil_image_to_data_url(image)
        return self._post_embedding(self._image_messages(image_data_url), index=index)

    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            dim = self.output_dim or 0
            return torch.empty((0, dim), dtype=torch.float32)
        max_workers = min(self.max_concurrency, len(images))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(self._post_image_embedding, enumerate(images)))
        return self._stack_embeddings(embeddings)

    @torch.no_grad()
    def encode_image_with_internal(self, images: list[Image.Image]):
        if not isinstance(images, list) or not images or not all(isinstance(image, Image.Image) for image in images):
            raise TypeError(
                f"Qwen vLLM backend expected a non-empty list of PIL images. Got {type(images).__name__}: {images!r}"
            )
        features = self._encode_image_batch(images)
        return features, features

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            dim = self.output_dim or 0
            return torch.empty((0, dim), dtype=torch.float32)
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError(f"Qwen vLLM backend expected a list of strings. Got {type(texts).__name__}: {texts!r}")
        messages = [self._text_messages(text) for text in texts]
        return self._encode_messages_batch(messages)

def safe_model_dir_name(value: str) -> str:
    return value.replace("/", "--").replace("\\", "--").replace(":", "_")


def create_search_model_meta_dir(args):
    if args.search_backend == "open_clip":
        model_dir_name = f"{args.search_model_name}-{args.search_model_pretrained}"
        metadata = SearchModelMetadata(
            model_name=args.search_model_name,
            pretrained=args.search_model_pretrained,
            display_name=model_dir_name,
        )
    else:
        model_dir_name = safe_model_dir_name(args.search_model_id)
        metadata = SearchModelMetadata(
            model_name=args.search_model_id,
            pretrained="",
            display_name=args.search_model_id,
        )
    search_model_meta_dir = pathlib.Path(args.meta_dir) / model_dir_name
    search_model_meta_dir.mkdir(parents=True, exist_ok=True)
    write_model_metadata(search_model_meta_dir, metadata)
    return str(search_model_meta_dir)


def configure_torch_backends():
    """Optimize CUDA/cuDNN backends for faster inference."""

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def search_embedding_model_label(args) -> str:
    if args.search_backend == "open_clip":
        return f"open_clip:{args.search_model_name}:{args.search_model_pretrained}"
    return f"{args.search_backend}:{args.search_model_id}"


def metadata_embedding_model_label(args) -> str | None:
    if args.disable_clip_metadata:
        return None
    return "open_clip:ViT-L-14:openai"


def load_metadata_embedding_backend(args, device) -> OpenClipEmbeddingBackend | None:
    if args.disable_clip_metadata:
        return None
    return OpenClipEmbeddingBackend("ViT-L-14", "openai", device)


def _to_device(batch, device):
    """Recursively move tensors and mapping-like processor outputs to a device."""

    if batch is None:
        return None
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: _to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_to_device(value, device) for value in batch]
    return batch


def load_search_embedding_backend(args, device) -> SearchEmbeddingBackend:
    if args.search_backend == "open_clip":
        return OpenClipEmbeddingBackend(
            args.search_model_name,
            args.search_model_pretrained,
            device,
        )
    if args.search_backend == "qwen_vl":
        if args.qwen_max_pixels is not None:
            print(
                "Warning: --qwen-max-pixels is ignored by the API backend.\n"
                "Configure max_pixels with vLLM --mm-processor-kwargs instead."
            )
        return QwenVlEmbeddingBackend(
            args.search_model_id,
            args.search_model_out_dim,
            args.qwen_api_base,
            args.qwen_api_key,
            args.qwen_api_timeout,
            args.qwen_api_max_retries,
            args.qwen_api_concurrency,
            args.qwen_instruction,
            args.qwen_api_skip_server_check,
        )
    raise ValueError(f"Unknown search backend: {args.search_backend}")


def create_image_id_index(file_list):
    image_id_to_path = {}
    path_to_image_id = {}
    for file in file_list:
        image_id = hashlib.sha256(str(file).encode()).hexdigest()
        image_id_to_path[image_id] = file
        path_to_image_id[file] = image_id
    return image_id_to_path, path_to_image_id


class ImageMetaFilterStats(typing.NamedTuple):
    valid_meta_count: int
    has_existing_metas: bool


class OrphanDbStats(typing.NamedTuple):
    total_db_count: int
    orphan_records: list[tuple[str, str | None]]

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_records)


def filter_valid_image_meta(args, result, image_id_to_path):
    valid_meta_count = 0
    has_existing_metas = False
    uncreated_image_paths = []
    expected_bytes = (
        args.search_model_out_dim * np.dtype(np.float32).itemsize
        if args.search_model_out_dim is not None
        else None
    )

    for image_id, meta in tqdm.tqdm(result, total=len(image_id_to_path)):
        if meta is not None:
            meta_view = memoryview(meta)
            if expected_bytes is None or meta_view.nbytes == expected_bytes:
                valid_meta_count += 1
                has_existing_metas = True
                continue

        uncreated_image_paths.append(image_id_to_path[image_id])

    return ImageMetaFilterStats(valid_meta_count, has_existing_metas), uncreated_image_paths


def infer_meta_dimension(con: sqlite3.Connection) -> int | None:
    """保存済みmeta BLOBからfloat32埋め込み次元を推定する。"""
    row = con.execute(
        "SELECT meta FROM image_meta WHERE meta IS NOT NULL LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    meta_size = memoryview(row[0]).nbytes
    item_size = np.dtype(np.float32).itemsize
    if meta_size % item_size != 0:
        raise ValueError(f"Invalid meta byte size for float32 vector: {meta_size}")
    return meta_size // item_size


def resolve_embedding_dimension(args, con: sqlite3.Connection) -> int:
    """CLI指定または保存済みmetaから現在の埋め込み次元を決定する。"""
    if args.search_model_out_dim is not None:
        return int(args.search_model_out_dim)
    inferred = infer_meta_dimension(con)
    if inferred is None:
        raise ValueError(
            "Embedding dimension is unknown because no embeddings were generated or stored."
        )
    args.search_model_out_dim = inferred
    print(f"Inferred search_model_out_dim={inferred}")
    return inferred


def collect_orphan_db_records(
    con: sqlite3.Connection, id_list: list[str], loop_size: int
) -> OrphanDbStats:
    total_db_count = int(con.execute("SELECT COUNT(*) FROM image_meta").fetchone()[0])
    if total_db_count == 0:
        return OrphanDbStats(total_db_count=0, orphan_records=[])

    cur = con.cursor()
    temp_table = f"valid_id_{uuid.uuid4().hex}"

    try:
        cur.execute(f'DROP TABLE IF EXISTS "{temp_table}"')
        cur.execute(f'CREATE TEMP TABLE "{temp_table}" (image_id TEXT PRIMARY KEY)')

        if id_list:
            cur.execute("BEGIN")
            try:
                for idx in tqdm.tqdm(range(0, len(id_list), loop_size)):
                    sub_list: list[str] = id_list[idx: idx + loop_size]
                    cur.executemany(
                        f'INSERT INTO "{temp_table}" (image_id) VALUES (?)',
                        ((image_id,) for image_id in sub_list),
                    )
            except Exception:
                con.rollback()
                raise
            else:
                con.commit()

        cur.execute(
            f"""
            SELECT image_meta.image_id, image_meta.image_path
              FROM image_meta
              LEFT JOIN "{temp_table}" AS valid_id
                ON image_meta.image_id = valid_id.image_id
             WHERE valid_id.image_id IS NULL
            """
        )
        orphan_records = list(cur.fetchall())
        return OrphanDbStats(total_db_count=total_db_count, orphan_records=orphan_records)
    finally:
        cur.execute(f'DROP TABLE IF EXISTS "{temp_table}"')


def print_image_meta_stats(meta_stats, uncreated_image_paths, max_len, orphan_db_stats):
    valid_count = meta_stats.valid_meta_count
    print(
        f"uncreated images:{len(uncreated_image_paths)}/{max_len} ({len(uncreated_image_paths)/max_len*100.:.4f}%)"
    )
    print(
        f"existing metas:{valid_count}/{max_len} ({valid_count/max_len*100.:.4f}%)"
    )
    orphan_ratio = (
        (orphan_db_stats.orphan_count / orphan_db_stats.total_db_count) * 100.0
        if orphan_db_stats.total_db_count
        else 0.0
    )
    print(
        f"orphan db records:{orphan_db_stats.orphan_count}/{orphan_db_stats.total_db_count} ({orphan_ratio:.4f}%)"
    )


def delete_orphan_db_records(con: sqlite3.Connection, orphan_db_stats: OrphanDbStats):
    if orphan_db_stats.orphan_count == 0:
        print("No orphan DB records to delete.")
        return

    print("Deleting orphan DB records...")
    with con:
        con.executemany(
            "DELETE FROM image_meta WHERE image_id = ?",
            ((image_id,) for image_id, _ in orphan_db_stats.orphan_records),
        )

    print("Deleted records:")
    for image_id, image_path in orphan_db_stats.orphan_records:
        print(f"- {image_path if image_path else image_id}")


def extract_image_features(
    args,
    device,
    search_model,
    con,
    cur,
    loader,
    uncreated_image_paths,
    uncreated_image_ids,
    aesthetic_model,
    pony_scorer,
    style_cluster,
    tagging_service,
    metadata_model=None,
):

    processed_count = 0
    processed_batches = 0
    commit_interval = max(1, 10000 // args.batch_size)

    with torch.inference_mode():
        for batch in tqdm.tqdm(loader, total=len(uncreated_image_paths) // args.batch_size + 1):
            if batch is None:
                continue

            (
                batched_image_input,
                batched_metadata_input,
                batched_image_index,
                batched_wd_inputs,
                batched_z3d_inputs,
            ) = batch

            batched_image_input = _to_device(batched_image_input, device)
            batched_metadata_input = _to_device(batched_metadata_input, device)
            try:
                (
                    _search_internal_features_tensor,
                    search_features_tensor,
                ) = search_model.encode_image_with_internal(batched_image_input)

                search_denom = search_features_tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                search_features_tensor = (search_features_tensor / search_denom).contiguous()

                batch_size = int(search_features_tensor.shape[0])
                needs_metadata_features = (
                    batched_metadata_input is not None
                    and metadata_model is not None
                    and (
                        aesthetic_model is not None
                        or pony_scorer is not None
                        or style_cluster is not None
                    )
                )
                aesthetic_scores = [None] * batch_size
                pony_scores = [None] * batch_size
                cluster_ids = [None] * batch_size

                if needs_metadata_features:
                    (
                        metadata_internal_features_tensor,
                        metadata_projected_features_tensor,
                    ) = metadata_model.encode_image_with_internal(batched_metadata_input)

                    metadata_features_for_scores = None
                    if aesthetic_model is not None or pony_scorer is not None:
                        # Projected OpenCLIP features are used by aesthetic/Pony scorers.
                        projected_norm = metadata_projected_features_tensor.norm(
                            dim=-1,
                            keepdim=True,
                        ).clamp_min(1e-12)
                        metadata_features_for_scores = (
                            metadata_projected_features_tensor / projected_norm
                        ).contiguous()

                    if aesthetic_model is not None:
                        aesthetic_dtype = next(aesthetic_model.parameters()).dtype
                        aesthetic_features = metadata_features_for_scores.to(dtype=aesthetic_dtype)
                        aesthetic_scores = (
                            aesthetic_model
                            .score_batch(aesthetic_features)
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )

                    if pony_scorer is not None:
                        pony_scores = pony_scorer.score_batch(metadata_features_for_scores)

                    if style_cluster is not None:
                        # Pre-projection/internal OpenCLIP features are used for style clustering.
                        style_features = metadata_internal_features_tensor.to(
                            device=style_cluster.cluster_centers.device,
                            dtype=style_cluster.cluster_centers.dtype,
                            non_blocking=True,
                        )
                        cluster_ids, _cluster_dists = style_cluster.get_cluster_batch(
                            style_features
                        )

                # 3) タグ付けもバッチで
                if args.use_existing_tags:
                    tagging_results = [
                        load_sidecar_tags(
                            args.image_dir,
                            uncreated_image_paths[int(idx)],
                        )
                        for idx in batched_image_index
                    ]
                else:
                    assert tagging_service is not None
                    wd_batch = (
                        batched_wd_inputs.cpu().numpy()
                        if isinstance(batched_wd_inputs, torch.Tensor)
                        else np.asarray(batched_wd_inputs)
                    )
                    z3d_batch = (
                        batched_z3d_inputs.cpu().numpy()
                        if isinstance(batched_z3d_inputs, torch.Tensor)
                        else np.asarray(batched_z3d_inputs)
                    )
                    tagging_results = tagging_service.tag_batch(
                        wd_batch,
                        z3d_batch,
                    )  # list[dict], len=B


                batched_new_search_meta = (
                    search_features_tensor.detach().cpu().float().numpy()
                )

                params = []
                time_stamp_ISO = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()

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
                    if args.search_model_out_dim is None:
                        args.search_model_out_dim = int(new_search_meta.shape[0])
                        print(f"Detected search_model_out_dim={args.search_model_out_dim}")
                    if new_search_meta.shape[0] != args.search_model_out_dim:
                        print(f"{new_search_meta.shape[0]} != {args.search_model_out_dim}")
                        continue
                    new_search_meta_bytes = new_search_meta.tobytes()
                    rating = tag_res.get("rating", "")
                    image_tags = ", ".join(tag_res.get("tags", []))
                    claster = f"style_cluster_{cluster_id}" if cluster_id is not None else ""

                    image_index_int = int(image_index)
                    image_path = uncreated_image_paths[image_index_int]
                    image_id = uncreated_image_ids[image_index_int]

                    params.append(
                        (
                            image_id,
                            image_path,
                            new_search_meta_bytes,
                            float(aesthetic_score) if aesthetic_score is not None else None,
                            float(pony_aesthetic_score) if pony_aesthetic_score is not None else None,
                            claster,
                            rating,
                            image_tags,
                            time_stamp_ISO,
                            search_embedding_model_label(args),
                            metadata_embedding_model_label(args),
                        )
                    )

                if params:
                    cur.executemany(
                        """insert into image_meta (
                                image_id,
                                image_path,
                                meta,
                                aesthetic_quality,
                                pony_aesthetic_quality,
                                style_cluster,
                                rating,
                                image_tags,
                                time_stamp_ISO,
                                search_embedding_model,
                                metadata_embedding_model
                            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            on conflict(image_id) do update set
                                meta=excluded.meta,
                                aesthetic_quality=excluded.aesthetic_quality,
                                pony_aesthetic_quality=excluded.pony_aesthetic_quality,
                                style_cluster=excluded.style_cluster,
                                rating=excluded.rating,
                                image_tags=excluded.image_tags,
                                time_stamp_ISO=excluded.time_stamp_ISO,
                                search_embedding_model=excluded.search_embedding_model,
                                metadata_embedding_model=excluded.metadata_embedding_model""",
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


def collect_train_samples_algL(
    con: sqlite3.Connection,
    d: int,
    train_samples: int,
    batch_size: int,
    total: int | None = None,
):
    """
    Algorithm L (Vitter 1985) による reservoir sampling（読み飛ばし式）。
    image_meta(meta BLOB, image_path TEXT) を image_path ORDER BY で走査し、
    次元 d の float32 ベクトルを k=min(train_samples, total) 件だけ均一サンプルする。
    """
    if total is None:
        total = con.execute("SELECT COUNT(*) FROM image_meta").fetchone()[0]

    k = min(train_samples, total)
    if k <= 0:
        return np.empty((0, d), dtype=np.float32)
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


class SQLiteMetaDataset(IterableDataset):
    def __init__(
        self,
        db_path: str,
        expected_bytes: int,
        min_rowid: int,
        max_rowid: int,
        fetch_size: int = 10000,
    ):

        self.db_path = db_path
        self.expected_bytes = expected_bytes
        self.min_rowid = min_rowid
        self.max_rowid = max_rowid
        self.fetch_size = fetch_size

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_row = self.min_rowid
            end_row = self.max_rowid

        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            total_range = self.max_rowid - self.min_rowid + 1
            chunk_size = math.ceil(total_range / num_workers)
            start_row = self.min_rowid + worker_id * chunk_size
            end_row = min(start_row + chunk_size - 1, self.max_rowid)

        con = sqlite3.connect(self.db_path, timeout=60.0)

        # --- 修正ポイント：Read Lockをこまめに解放するページネーション ---

        current_min = start_row
        while current_min <= end_row:
            # fetch_size分だけ読み込んで、すぐにデータを取得

            cur = con.execute(
                """
                SELECT rowid, image_id, meta
                FROM image_meta
                WHERE rowid BETWEEN ? AND ?
                  AND meta IS NOT NULL
                ORDER BY rowid
                LIMIT ?
                """,
                (current_min, end_row, self.fetch_size),
            )

            rows = cur.fetchall()
            cur.close()

            if not rows:
                break

            for rowid, image_id, meta_blob in rows:
                current_min = rowid + 1  # 次回の開始位置を更新

                if len(meta_blob) != self.expected_bytes:
                    yield None, None
                    continue

                vec = np.frombuffer(meta_blob, dtype=np.float32).copy()
                yield image_id, torch.from_numpy(vec)

        con.close()


def _meta_collate_fn(batch):
    """バッチ内の None を除外し、テンソル化と処理件数を返す"""

    valid_ids = []
    vecs = []
    total_count = len(batch)

    for item in batch:
        if item[0] is not None:
            valid_ids.append(item[0])
            vecs.append(item[1])

    if not vecs:
        return [], torch.empty(0), total_count

    return valid_ids, torch.stack(vecs), total_count


def recalc_aesthetic_scores(
    con: sqlite3.Connection,
    aesthetic_model: nn.Module,
    pony_scorer: "PonyAestheticScorer",
    device: str,
    out_dim: int,
    batch_size: int = 16384,
    num_workers: int = 8,  # 追加: DataLoaderのワーカー数
) -> int:
    """
    DataLoaderを使ってメタデータの読み込みとテンソル変換を並列化して再計算を行う。
    """
    # 現在のコネクションからDBの物理ファイルパスを取得 (インメモリDBの場合は機能しません)
    print("[recalc_aesthetic] checking database file path...")
    db_list = con.execute("PRAGMA database_list").fetchall()
    db_path = db_list[0][2] if len(db_list) > 0 else ""
    if not db_path:
        raise ValueError(
            "[recalc_aesthetic] DataLoader parallelization requires a physical database file."
        )
    expected_bytes = out_dim * np.dtype(np.float32).itemsize
    print("[recalc_aesthetic] counting rows...")
    total = con.execute(
        "SELECT COUNT(*) FROM image_meta"
    ).fetchone()[0]
    print(f"[recalc_aesthetic] finding rowid range...")
    rowid_range = con.execute(
        "SELECT MIN(rowid), MAX(rowid) FROM image_meta"
    ).fetchone()
    min_rowid, max_rowid = rowid_range if rowid_range[0] is not None else (0, -1)
    print(f"[recalc_aesthetic] target rows with meta: {total} (workers: {num_workers})")
    if total == 0 or min_rowid > max_rowid:
        return 0

    dataset = SQLiteMetaDataset(
        db_path, expected_bytes, min_rowid, max_rowid, fetch_size=batch_size * 4
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_meta_collate_fn,
        prefetch_factor=16 if num_workers > 0 else None,
    )

    updated = 0
    pbar = tqdm.tqdm(total=total, unit="row", desc="recalc_aesthetic")
    aesthetic_dtype = next(aesthetic_model.parameters()).dtype

    # DataLoader からバッチを受け取って推論
    for valid_ids, feats, processed_count in dataloader:
        pbar.update(processed_count)
        if not valid_ids:
            continue

        # --- バッチ推論 ---
        feats = feats.to(device=device, dtype=aesthetic_dtype)
        # L2正規化
        denom = feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats_norm = feats / denom
        with torch.no_grad():
            aesthetic_scores: np.ndarray = (
                aesthetic_model.score_batch(feats_norm).squeeze(-1).cpu().numpy()
            )
            pony_scores: np.ndarray = pony_scorer.score_batch(feats_norm)
        # --- DB 更新 ---
        params = [
            (float(a), float(p), iid)
            for iid, a, p in zip(valid_ids, aesthetic_scores, pony_scores)
        ]

        con.executemany(
            """

            UPDATE image_meta
               SET aesthetic_quality      = ?,
                   pony_aesthetic_quality = ?
             WHERE image_id = ?
            """,
            params,
        )

        con.commit()
        # 最適化ルーチンのトリガーチェック (100万件更新ごとに実行)
        prev_updated = updated
        updated += len(params)

        if (updated // 1_000_000) > (prev_updated // 1_000_000):
            print(
                f"\n[recalc_aesthetic] updated {updated} rows so far, running optimize..."
            )
            res = con.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
            # res は (busy, log, checkpointed) のタプル。busy=1 なら競合して失敗している
            print(f"\n[recalc_aesthetic] checkpoint result (busy, log, chkpt): {res}")
            con.execute("PRAGMA vacuum;")
            con.execute("PRAGMA optimize;")

    pbar.close()
    print(f"[recalc_aesthetic] updated {updated} rows")
    return updated


def compute_mean_vector(
    con: sqlite3.Connection, d: int, batch_size: int = 50_000
) -> np.ndarray:
    """全metaベクトルをL2正規化してから平均ベクトルを計算する。
    ビルド/検索側で `faiss.normalize_L2` を経たベクトルと整合させるため、
    ここでも明示的に正規化してから平均を取る。
    """
    print("Computing mean vector...")
    cur = con.execute("SELECT meta FROM image_meta WHERE meta IS NOT NULL")
    mean_vec = np.zeros(d, dtype=np.float64)
    total_count = 0
    pbar = tqdm.tqdm(desc="mean computation")

    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        batch_arr = []
        for (meta_blob,) in rows:
            a = np.frombuffer(meta_blob, dtype=np.float32)
            if a.size == d:
                batch_arr.append(a)

        if not batch_arr:
            continue

        # 連続メモリの float32 配列にしてから正規化(faiss.normalize_L2 の要件)
        batch_np = np.ascontiguousarray(np.stack(batch_arr), dtype=np.float32)
        faiss.normalize_L2(batch_np)              # ← 追加

        # float64 で累積して精度を保つ
        mean_vec += batch_np.sum(axis=0, dtype=np.float64)
        total_count += batch_np.shape[0]
        pbar.update(len(batch_arr))

    pbar.close()

    if total_count == 0:
        raise ValueError("No valid meta vectors found.")

    return (mean_vec / total_count).astype(np.float32)


def stream_build_faiss(
    db_path: str,
    nlist: int, pq_m: int, bits_per_code: int,
    d: int,
    out_index_path: str,
    batch_size: int = 50_000,
    train_samples: int = 2_000_000,
):
    con = connect_db(db_path)
    try:
        con.execute("PRAGMA case_sensitive_like=OFF")
        con.execute("PRAGMA temp_store=MEMORY")

        total = con.execute(
            "SELECT COUNT(*) FROM image_meta WHERE meta IS NOT NULL"
        ).fetchone()[0]
        print(f"total rows with meta: {total}")

        if total == 0:
            print("No valid meta found, skipping index creation.")
            return

        # -------- 平均ベクトルの計算と保存 --------
        mean_vec = compute_mean_vector(con, d, batch_size).astype(np.float32)
        mean_vec_path = out_index_path + ".mean.npy"
        np.save(mean_vec_path, mean_vec)
        print(f"Saved mean vector to {mean_vec_path}")

        metadata_path = out_index_path + ".meta.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"embedding_dim": d, "pq_m": pq_m}, f, ensure_ascii=False, indent=2)
        print(f"Saved index metadata to {metadata_path}")

        # -------- パス1: 訓練 --------
        train_buf = collect_train_samples_algL(con, d, train_samples, batch_size, total=total)
        train_buf = np.ascontiguousarray(train_buf, dtype=np.float32)

        # ① 正規化 → ② 中心化(再正規化しない)
        faiss.normalize_L2(train_buf)
        train_buf -= mean_vec

        index = createIndex(nlist, pq_m, bits_per_code, d)
        assert index.metric_type == faiss.METRIC_INNER_PRODUCT, \
            f"Index must be IP metric, got {index.metric_type}"

        print(f"train {train_buf.shape}")
        index.train(train_buf)
        del train_buf

        # -------- パス2: 逐次add --------
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
                break  # 各バッチで即 add しているので、ここでの flush は不要

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
                # ① 正規化 → ② 中心化(再正規化しない)
                faiss.normalize_L2(buf[:k])
                buf[:k] -= mean_vec
                index.add(buf[:k])

        pbar.close()
        print("write")
        faiss.write_index(index, out_index_path)
    finally:
        print("close db connection")
        con.close()


@torch.no_grad()
def main():

    # 起動引数
    args = parse_arguments()

    configure_torch_backends()

    # 作業フォルダの作成
    search_model_meta_dir: str = create_search_model_meta_dir(args)

    # --recalc-aesthetic モード: 画像スキャン不要。DB の meta ベクトルだけを再利用する。
    if getattr(args, "recalc_aesthetic", False):
        print("[recalc_aesthetic] mode: skipping image scan / CLIP inference / tagging")
        configure_torch_backends()
        device = get_device()

        # aesthetic モデルだけ読み込む（CLIP・タガーは不要）
        aesthetic_model: Aesthetic_model = get_aesthetic_model(args.aesthetic_model_path, clip_model="vit_l_14")
        aesthetic_model = aesthetic_model.to(device)
        aesthetic_model.eval()

        aesthetic_checkpoint_arg = getattr(args, "aesthetic_checkpoint", None)
        aesthetic_checkpoint = ensure_asset_path(
            aesthetic_checkpoint_arg,
            AESTHETIC_CHECKPOINT_FILENAME,
            "aesthetic checkpoint",
        )
        pony_scorer = PonyAestheticScorer(device=device, checkpoint=aesthetic_checkpoint)

        con = connect_db(search_model_meta_dir)
        try:
            cur = con.cursor()
            init_db(cur)

            recalc_batch = getattr(args, "recalc_aesthetic_batch_size", 16384)
            embedding_dim = resolve_embedding_dimension(args, con)
            recalc_aesthetic_scores(
                con,
                aesthetic_model,
                pony_scorer,
                device,
                out_dim=embedding_dim,
                batch_size=recalc_batch,
            )

            cur.execute("pragma optimize")
        finally:
            print("close db connection")
            con.close()

        # FAISSインデックスを再構築
        stream_build_faiss(
            search_model_meta_dir,
            args.nlist,
            args.pq_m,
            args.bits_per_code,
            embedding_dim,
            f"{search_model_meta_dir}/{args.metas_faiss_index_file_name}",
            batch_size=100_000,
            train_samples=2_000_000,
        )
        print("[recalc_aesthetic] done.")
        return

    # 画像ファイル一覧を求める
    print("files")
    dir_path: pathlib.Path = pathlib.Path(args.image_dir)
    print(dir_path)

    ext_list = get_default_image_extensions()

    print(ext_list)

    index_item_list = get_image_list_from_dir(
        dir_path,
        exts=ext_list,
        scan_timeout_sec=args.scan_timeout_sec,
    )
    print(len(index_item_list))

    # データベースの準備
    print("load_db")
    con: sqlite3.Connection = connect_db(search_model_meta_dir)
    try:
        cur: sqlite3.Cursor = con.cursor()
        init_db(cur)

        # GPUが使用可能か
        device = get_device()

        # モデルの読み込み
        print("load_model")

        search_model = load_search_embedding_backend(args, device)
        metadata_model = load_metadata_embedding_backend(args, device)
        eval_transform = search_model.preprocess
        metadata_transform = metadata_model.preprocess if metadata_model is not None else None

        if args.disable_clip_metadata:
            aesthetic_model = None
            pony_scorer = None
            style_cluster = None
        else:
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
        if args.use_existing_tags:
            tagging_service = None
            wd_target_size = 448  # タグ付けを行わない場合でも前処理の形を維持
        else:
            tagging_service = ImageTaggingService(hf_token=None)
            wd_target_size = tagging_service.ensure_wd_model_loaded()

        index_item_list, path_to_image_id = create_image_id_index(index_item_list)

        # データベースに問い合わせる
        result = load_image_meta_from_db(con, id_list=list(index_item_list.keys()), loop_size=10000)

        meta_stats, uncreated_image_paths = filter_valid_image_meta(args, result, index_item_list)
        uncreated_image_ids = [path_to_image_id[path] for path in uncreated_image_paths]
        orphan_db_stats = collect_orphan_db_records(
            con,
            id_list=list(index_item_list.keys()),
            loop_size=10000,
        )

        print_image_meta_stats(meta_stats, uncreated_image_paths, len(index_item_list), orphan_db_stats)

        if args.delete_orphan_db_records:
            delete_orphan_db_records(con, orphan_db_stats)

        del result, index_item_list, meta_stats, orphan_db_stats

        dataset = None
        T = True
        i = 0
        loader = None
        if not uncreated_image_paths:
            print("No new images to process.")
        else:

            while T:
                if i < 1:
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
                else:
                    num_workers = max(0, args.num_workers-2*i)
                    dataloader_kwargs = dict(
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=False,
                        collate_fn=safe_collate,
                    )

                target_paths = uncreated_image_paths[i * args.batch_size: ]
                target_ids = uncreated_image_ids[i * args.batch_size: ]
                if not target_paths:
                    break

                dataset = DirImageIterable(
                    args.image_dir,
                    target_paths,
                    transform=eval_transform,
                    wd_target_size=wd_target_size,
                    metadata_transform=metadata_transform,
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
                        target_ids,
                        aesthetic_model,
                        pony_scorer,
                        style_cluster,
                        tagging_service,
                        metadata_model,
                    )
                    T = False
                except Exception:
                    traceback.print_exc()
                    i += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

        con.commit()

        del uncreated_image_paths, uncreated_image_ids, search_model, dataset, loader
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
        con.execute("pragma wal_checkpoint(TRUNCATE);")
        print("pragma vacuum")
        cur.execute("pragma vacuum")
        print("pragma optimize")
        cur.execute("pragma optimize")

    finally:
        print("close db connection")
        con.close()

    del con, cur

    con = connect_db(search_model_meta_dir)
    try:
        embedding_dim = resolve_embedding_dimension(args, con)
    finally:
        con.close()

    stream_build_faiss(search_model_meta_dir, args.nlist, args.pq_m, args.bits_per_code, embedding_dim,
        f"{search_model_meta_dir}/{args.metas_faiss_index_file_name}", batch_size=100_000, train_samples=2_000_000
    )


if __name__ == "__main__":
    main()
