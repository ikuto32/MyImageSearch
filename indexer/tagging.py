"""画像のタグとratingを求める。ONNXのタガー2種、または画像の隣のタグファイルを使う。"""
from __future__ import annotations

import csv
import json
import pathlib

import huggingface_hub
import numpy as np
import onnxruntime as rt
from PIL import Image

WD_TAGGER_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
Z3D_TAGGER_REPO = "toynya/Z3D-E621-Convnext"
Z3D_INPUT_SIZE = 448

# 下線を空白に置き換えないタグ。
KAOMOJIS = frozenset([
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
])

# selected_tags.csv / tags-selected.csv のcategory列。
WD_RATING, WD_GENERAL, WD_CHARACTER = 9, 0, 4
Z3D_GENERAL = ("0", "3", "5")
Z3D_CHARACTER = "4"


def prepare_tag_input(image: Image.Image, target_size: int) -> np.ndarray:
    """
    Tagging 共通の前処理を行う純関数.

    RGB化 → 縦横比維持のリサイズ → 白背景へのパディング → BGR/float32 へ変換.
    バッチ次元は付与しない (H, W, 3) を返す。
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    resized_image = image.resize(new_size, Image.LANCZOS)

    padded_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    padded_image.paste(resized_image, (
        (target_size - new_size[0]) // 2,
        (target_size - new_size[1]) // 2,
    ))

    # Channel reversal introduces negative strides; copy to keep collate happy.
    return np.asarray(padded_image, dtype=np.float32)[:, :, ::-1].copy()  # RGB → BGR


def _create_session(model_path: str) -> rt.InferenceSession:
    options = rt.SessionOptions()
    options.log_severity_level = 2
    options.log_verbosity_level = 0
    return rt.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        sess_options=options,
    )


def _names_above(names: np.ndarray, indexes: np.ndarray, scores: np.ndarray, threshold: float) -> list[str]:
    # ラベルの行数がモデルの出力より多い場合、出力のないラベルは対象にしない。
    indexes = indexes[indexes < scores.shape[0]]
    return names[indexes[scores[indexes] > threshold]].tolist()


class WdTagger:
    """SmilingWolf/wd-swinv2 ベースのタグ付け。"""

    def __init__(self, model_repo: str = WD_TAGGER_REPO):
        csv_path = huggingface_hub.hf_hub_download(model_repo, "selected_tags.csv")
        model_path = huggingface_hub.hf_hub_download(model_repo, "model.onnx")
        with open(csv_path, encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self._load_labels(rows)
        self.session = _create_session(model_path)
        _, height, _width, _ = self.session.get_inputs()[0].shape
        self.input_size = int(height)

    def _load_labels(self, rows: list[dict]) -> None:
        self.names = np.array([
            row["name"] if row["name"] in KAOMOJIS else row["name"].replace("_", " ") for row in rows
        ], dtype=object)
        categories = np.array([int(row["category"]) for row in rows])
        self.rating_indexes = np.flatnonzero(categories == WD_RATING)
        self.general_indexes = np.flatnonzero(categories == WD_GENERAL)
        self.character_indexes = np.flatnonzero(categories == WD_CHARACTER)

    def predict_batch(
        self, images: np.ndarray, *, general_threshold: float, character_threshold: float,
    ) -> list[tuple[str, list[str], list[str]]]:
        """前処理済みの(N, H, W, 3)から、画像ごとの(rating, 一般タグ, キャラクタータグ)を返す。"""
        if images.size == 0:
            return []
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        predictions = self.session.run([output_name], {input_name: images})[0]
        # しきい値との比較はfloat64で行う(float32に丸めたしきい値では境界の値の扱いが変わる)。
        return [
            self._labels(scores, general_threshold, character_threshold)
            for scores in np.asarray(predictions, dtype=np.float64)
        ]

    def _labels(self, scores: np.ndarray, general_threshold: float, character_threshold: float):
        rating = ""
        if self.rating_indexes.size:
            rating = self.names[self.rating_indexes[np.argmax(scores[self.rating_indexes])]]
        return (
            rating,
            _names_above(self.names, self.general_indexes, scores, general_threshold),
            _names_above(self.names, self.character_indexes, scores, character_threshold),
        )


class Z3DTagger:
    """toynya/Z3D-E621-Convnext ベースのタグ付け。"""

    OUTPUT_NAME = "predictions_sigmoid"

    def __init__(self, model_repo: str = Z3D_TAGGER_REPO, threshold: float = 0.5):
        self.threshold = threshold
        csv_path = huggingface_hub.hf_hub_download(model_repo, "tags-selected.csv")
        model_path = huggingface_hub.hf_hub_download(model_repo, "model.onnx")
        with open(csv_path, encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self._load_labels(rows)
        self.session = _create_session(model_path)

    def _load_labels(self, rows: list[dict]) -> None:
        self.names = np.array([row["name"].strip().replace("_", " ") for row in rows], dtype=object)
        categories = np.array([row["category"] for row in rows], dtype=object)
        self.general_indexes = np.flatnonzero(np.isin(categories, Z3D_GENERAL))
        self.character_indexes = np.flatnonzero(categories == Z3D_CHARACTER)

    def predict_batch(self, images: np.ndarray) -> list[tuple[list[str], list[str]]]:
        """前処理済みの(N, H, W, 3)から、画像ごとの(一般タグ, キャラクタータグ)を返す。"""
        if images.size == 0:
            return []
        model_input = self.session.get_inputs()[0]
        images = images.astype(np.float32, copy=False)
        if model_input.shape[0] == 1:
            # バッチ次元が1に固定されたモデルは、1枚ずつ推論する。
            predictions = [
                self.session.run([self.OUTPUT_NAME], {model_input.name: image[np.newaxis]})[0][0]
                for image in images
            ]
        else:
            predictions = self.session.run([self.OUTPUT_NAME], {model_input.name: images})[0]
        return [self._labels(scores) for scores in np.asarray(predictions, dtype=np.float64)]

    def _labels(self, scores: np.ndarray):
        return (
            _names_above(self.names, self.general_indexes, scores, self.threshold),
            _names_above(self.names, self.character_indexes, scores, self.threshold),
        )


def _merge(first: list[str], second: list[str]) -> list[str]:
    """重複を除き、最初に現れた順序を保つ。"""
    return list(dict.fromkeys((*first, *second)))


class ImageTaggingService:
    """2つのタガー(WD + Z3D)の結果をまとめ、ratingとタグの一覧にする。"""

    def __init__(self, general_threshold: float = 0.35, character_threshold: float = 0.85):
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.wd_tagger = WdTagger()
        self.z3d_tagger = Z3DTagger()

    @property
    def input_sizes(self) -> tuple[int, int]:
        """(WD, Z3D)の順の、前処理後の画像の一辺。"""
        return self.wd_tagger.input_size, Z3D_INPUT_SIZE

    def tag_batch(self, wd_batch: np.ndarray, z3d_batch: np.ndarray) -> list[dict]:
        wd_results = self.wd_tagger.predict_batch(
            wd_batch,
            general_threshold=self.general_threshold,
            character_threshold=self.character_threshold,
        )
        z3d_results = self.z3d_tagger.predict_batch(z3d_batch)
        return [
            {
                "rating": rating,
                "tags": _merge(general, z3d_general)
                + [f"character:{tag}" for tag in _merge(characters, z3d_characters)],
            }
            for (rating, general, characters), (z3d_general, z3d_characters) in zip(wd_results, z3d_results)
        ]


def load_sidecar_tags(image_dir: str, image_path: str) -> dict:
    """Read photo.jpg.tags.json next to an image, or leave it unclassified."""

    image = pathlib.Path(image_dir) / image_path
    sidecar = image.with_name(image.name + ".tags.json")
    empty = {"rating": "", "tags": []}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("expected an object with rating and tags")
        rating = data.get("rating", "")
        tags = data.get("tags", [])
        if not isinstance(rating, str) or not isinstance(tags, list) or any(
            not isinstance(tag, str) for tag in tags
        ):
            raise ValueError("rating must be a string and tags must be a list of strings")
        return {"rating": rating, "tags": tags}
    except FileNotFoundError:
        return empty
    except (OSError, UnicodeError, ValueError) as error:
        print(f"[tags warning] {sidecar}: {error}; using empty tags and no rating")
        return empty
