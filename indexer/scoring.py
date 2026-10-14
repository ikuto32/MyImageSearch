"""OpenCLIP ViT-L/14の特徴量から審美性評価とスタイルクラスタを求めるモデル。"""
from __future__ import annotations

import os

import huggingface_hub
import numpy as np
import torch
import torch.nn as nn

AESTHETIC_REPO = "purplesmartai/aesthetic-classifier"
AESTHETIC_CHECKPOINT_FILENAME = "v2.ckpt"
STYLE_REPO = "purplesmartai/style-classifier"
STYLE_CENTERS_FILENAME = "clustering_results_n2048_gpu.npz"

# 審美性評価とスタイルクラスタが前提とするOpenCLIPモデル。
METADATA_MODEL_NAME = "ViT-L-14"
METADATA_MODEL_PRETRAINED = "openai"


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


def _l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class AestheticModel(nn.Module):
    """射影後のCLIP特徴量から審美性評価を求める。層の名前は学習済み重みのキーに対応する。"""

    def __init__(self, input_dim=768):
        super().__init__()
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

    @torch.no_grad()
    def score_batch(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, D) → (N,) の美的評価スコアを返す"""
        return self.forward(_l2_normalize(features)).squeeze(-1).clamp(0, 1)


def load_aesthetic_model(path: str, device: str) -> AestheticModel:
    model = AestheticModel()
    raw = torch.load(path, map_location="cpu")
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


class PonyAestheticScorer:
    """purplesmartai/aesthetic-classifier による審美性評価。"""

    def __init__(self, checkpoint: str, device: str, input_size: int = 768):
        self.device = device
        self.aesthetic_model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        ).to(self.device)
        # 読み込みに失敗した重みで評価を保存しないよう、失敗した場合は処理を止める。
        try:
            checkpoint_data = torch.load(checkpoint, map_location=self.device)
            state_dict = checkpoint_data.get("state_dict", checkpoint_data)
            self.aesthetic_model.load_state_dict({
                key.replace("model.", ""): value for key, value in state_dict.items()
            })
        except Exception as error:
            raise RuntimeError(
                f"Could not load the aesthetic checkpoint {checkpoint}. "
                "Fix the file, or index without aesthetic/style metadata using --disable-clip-metadata."
            ) from error
        self.aesthetic_model.eval()

    @torch.no_grad()
    def score_batch(self, features: torch.Tensor | np.ndarray) -> np.ndarray:
        """features: (N, D) → (N,) の評価値"""
        if not torch.is_tensor(features):
            features = torch.from_numpy(features)
        features = _l2_normalize(features.to(self.device))
        return self.aesthetic_model(features).squeeze(-1).cpu().numpy()


# 不要なクラスタ(割り当てから除外する)
BLOCKLIST_CLUSTERS = frozenset([
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
    """射影前のCLIP特徴量を、最も近いスタイルクラスタ中心に割り当てる。"""

    def __init__(self, checkpoint_centers: str, device: str):
        try:
            with np.load(checkpoint_centers, allow_pickle=True) as data:
                original_centers = data["cluster_centers"]
        except Exception as error:
            raise RuntimeError(
                f"Could not load the style cluster centers {checkpoint_centers}. "
                "Fix the file, or index without aesthetic/style metadata using --disable-clip-metadata."
            ) from error
        self.kept_cluster_indices = [
            index for index in range(original_centers.shape[0]) if index not in BLOCKLIST_CLUSTERS
        ]
        print(f"Style cluster centers: {len(self.kept_cluster_indices)}/{original_centers.shape[0]} after blocklist")
        self.cluster_centers = torch.tensor(
            original_centers[self.kept_cluster_indices], dtype=torch.float32
        ).to(device)

    def _prepare_style_features(self, features: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(features):
            features = torch.from_numpy(features)
        if features.ndim != 2:
            raise ValueError(f"Style features must have shape (N, D), got {tuple(features.shape)}")
        expected_dim = int(self.cluster_centers.shape[1])
        actual_dim = int(features.shape[1])
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
    def get_cluster_batch(self, features: torch.Tensor | np.ndarray) -> tuple[list[int], np.ndarray]:
        """features: (N, D) → (元のクラスタID, 中心までの距離)"""
        features = self._prepare_style_features(features)
        distances = torch.cdist(features, self.cluster_centers)  # (N, C)
        min_dists, min_idx = distances.min(dim=1)
        cluster_ids = [self.kept_cluster_indices[int(index)] for index in min_idx.cpu().tolist()]
        return cluster_ids, min_dists.cpu().numpy()
