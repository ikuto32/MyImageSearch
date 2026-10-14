"""画像を読み込み、検索・メタデータ・タグ付けの各モデル向けに前処理するデータセット。"""
from __future__ import annotations

import os
from typing import Any

from blake3 import blake3
from PIL import Image, ImageFile
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data._utils.collate import default_collate

from indexer.tagging import prepare_tag_input

# DataLoaderのワーカーもこのモジュールを読み込むため、ここで設定する。
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 268_435_456
# ハッシュを求めるときに一度に読む量。ファイル全体をメモリに載せない。
_HASH_CHUNK_BYTES = 1024 * 1024


def open_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA").convert("RGB")


def _file_identity(path: str) -> tuple[int, int, int]:
    status = os.stat(path)
    return status.st_ino, status.st_size, status.st_mtime_ns


def open_rgb_with_hash(path: str) -> tuple[Image.Image, str]:
    """画像を開き、ファイルの内容のblake3ハッシュ(16進)と合わせて返す。

    画像はハッシュを記録する前と同じくパスから読む。読めた画像だけを少しずつ読んでハッシュするため、
    画像でないファイルは先頭だけで失敗し、大きなファイルも一度にメモリへ載せない。
    読み込みの途中でファイルが書き換えられた場合は、画像とハッシュが食い違わないよう失敗とする。
    """
    before = _file_identity(path)
    image = open_rgb(path)
    hasher = blake3()
    chunk = bytearray(_HASH_CHUNK_BYTES)
    view = memoryview(chunk)
    with open(path, "rb") as stream:
        while size := stream.readinto(chunk):
            hasher.update(view[:size])
    if _file_identity(path) != before:
        raise OSError(f"The file changed while it was being read: {path}")
    return image, hasher.hexdigest()


class DirImageIterable(IterableDataset):
    """(検索入力, メタデータ入力, 画像の番号, WDタガー入力, Z3Dタガー入力, blake3ハッシュ)を順に返す。

    使わないモデルの入力はNoneになる。読めない画像は記録して飛ばす。
    """

    def __init__(
        self,
        images_dir: str,
        img_list,
        transform=None,
        *,
        tag_sizes: tuple[int, int] | None,
        metadata_transform=None,
    ):
        self.images_dir = images_dir
        self.img_list = list(img_list)
        self.transform = transform
        self.metadata_transform = metadata_transform
        # (WD, Z3D)のタガー入力の一辺。Noneならタグ付け用の前処理を行わない。
        self.tag_sizes = tag_sizes

    def __getitem__(self, index: int):
        """Iterable datasets do not support random access."""
        raise TypeError(f"{self.__class__.__name__} does not support indexing")

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            indexes = range(len(self.img_list))
        else:
            # 各ワーカーに均等分割
            per_worker = (len(self.img_list) + info.num_workers - 1) // info.num_workers
            start = info.id * per_worker
            indexes = range(start, min(start + per_worker, len(self.img_list)))
            print(f"[DirImageIterable] worker {info.id} processing {indexes.start} to {indexes.stop} of {len(self.img_list)}")

        for index in indexes:
            path = os.path.join(self.images_dir, self.img_list[index])
            try:
                sample = self._load(path, index)
            except Exception as error:
                # 1枚の失敗でワーカー全体を止めない。
                print(f"[DirImageIterable] skipped: {path} -> {error}")
                continue
            yield sample

    def _load(self, path: str, index: int):
        image, file_hash = open_rgb_with_hash(path)
        wd_input = z3d_input = None
        if self.tag_sizes is not None:
            wd_size, z3d_size = self.tag_sizes
            wd_input = prepare_tag_input(image, wd_size)
            z3d_input = wd_input if z3d_size == wd_size else prepare_tag_input(image, z3d_size)
        search_input = image if self.transform is None else self.transform(image)
        metadata_input = None if self.metadata_transform is None else self.metadata_transform(image)
        return search_input, metadata_input, index, wd_input, z3d_input, file_hash


def _collate_search_inputs(search_inputs: list[Any]):
    # PIL画像(APIバックエンド用)はテンソルにまとめず、リストのまま渡す。
    if search_inputs and all(isinstance(item, Image.Image) for item in search_inputs):
        return search_inputs
    return default_collate(search_inputs)


def _collate_optional(values: list[Any]):
    if all(value is None for value in values):
        return None
    return default_collate(values)


def safe_collate(batch):
    """Collate search, metadata, index and tagger inputs, and the file hashes, safely."""

    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    search_inputs, metadata_inputs, indices, wd_inputs, z3d_inputs, file_hashes = zip(*batch)

    return (
        _collate_search_inputs(list(search_inputs)),
        _collate_optional(list(metadata_inputs)),
        default_collate(list(indices)),
        _collate_optional(list(wd_inputs)),
        _collate_optional(list(z3d_inputs)),
        list(file_hashes),
    )
