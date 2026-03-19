import asyncio
import contextlib
from dataclasses import dataclass
from functools import cache
import hashlib
import io
import itertools
import logging
import os
import pathlib
import mimetypes
import threading
import time
from typing import List, Sequence
import concurrent.futures
import zipfile

import tqdm

from app.domain.domain_object import (
    ImageItem,
    ImageId,
    Image,
    ImageName,
    ImageTags,
    ModelId,
    ModelItem,
    ModelName,
)
from app.domain.repository import Repository

from PIL import Image as PILImage

import open_clip

@dataclass
class _ScanProgress:
    started_at: float
    last_progress_at: float


class LocalRepository(Repository):
    """ローカル上のファイルを対象としたRepository"""

    _PARALLEL = 64
    _SCAN_TIMEOUT_SEC = 3600.0
    _SCAN_STALL_LOG_INTERVAL_SEC = 300.0
    _IMAGE_ITEM_WORKERS = 32

    def __init__(
        self,
        image_dir_path: pathlib.Path,
        *,
        scan_timeout_sec: float = _SCAN_TIMEOUT_SEC,
        scan_parallelism: int = _PARALLEL,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._image_dir_path: pathlib.Path = image_dir_path
        self._id_to_path: dict[ImageId, pathlib.Path] = {}
        self._scan_timeout_sec = scan_timeout_sec
        self._scan_parallelism = scan_parallelism

    @staticmethod
    def _get_default_image_extensions() -> list[str]:
        """Pillow が現在サポートしている拡張子一覧を返す。"""
        return sorted(PILImage.registered_extensions().keys())

    def _run_coro_sync(self, coro):
        """
        同期メソッドから安全に coroutine を実行する。

        - 通常の同期コンテキストでは asyncio.run() を使う
        - すでに event loop が動作中なら別スレッドで実行する
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = None
        error = None

        def runner():
            nonlocal result, error
            try:
                result = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
                error = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if error is not None:
            raise error
        return result

    async def _collect_image_relative_paths(
        self,
        exts: Sequence[str],
    ) -> list[pathlib.Path]:
        """
        create_index.py の get_image_list_from_dir/_collect_images 相当。

        - Pillow の拡張子一覧を利用
        - ディレクトリごとに ThreadPoolExecutor で scandir
        - 停滞したスキャンだけ timeout でスキップ
        - 返り値は root からの相対パス
        """
        root = self._image_dir_path
        loop = asyncio.get_running_loop()

        ext_set = {
            ("." + ext if not ext.startswith(".") else ext).lower()
            for ext in exts
        }
        ext_tuple = tuple(ext_set)

        files: list[pathlib.Path] = []
        progress_bar = tqdm.tqdm(
            unit="file",
            dynamic_ncols=True,
            desc="画像を走査中",
        )

        in_progress_scans: dict[pathlib.Path, _ScanProgress] = {}
        in_progress_lock = asyncio.Lock()
        thread_progress_lock = threading.Lock()
        semaphore = asyncio.Semaphore(self._scan_parallelism)

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

        def _scan_dir_sync(
            path: pathlib.Path,
        ) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
            dirs: list[pathlib.Path] = []
            hits: list[pathlib.Path] = []

            try:
                with os.scandir(path) as it:
                    for entry in it:
                        _mark_progress(path)
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                dirs.append(pathlib.Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                name_lc = entry.name.lower()
                                if name_lc.endswith(ext_tuple):
                                    hits.append(
                                        pathlib.Path(
                                            os.path.relpath(entry.path, root)
                                        )
                                    )
                        except OSError:
                            # 個別エントリのアクセス失敗は無視
                            continue
            except (PermissionError, FileNotFoundError, NotADirectoryError):
                # create_index.py と同様、読めないディレクトリはスキップ
                pass

            return dirs, hits

        async def _scan_with_timeout(
            path: pathlib.Path,
        ) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
            start = time.monotonic()
            async with in_progress_lock:
                in_progress_scans[path] = _ScanProgress(
                    started_at=start,
                    last_progress_at=start,
                )

            future = loop.run_in_executor(pool, _scan_dir_sync, path)
            try:
                while True:
                    try:
                        return await asyncio.wait_for(
                            asyncio.shield(future),
                            timeout=self._scan_timeout_sec,
                        )
                    except TimeoutError:
                        snapshot = _get_progress_snapshot(path)
                        if snapshot is None:
                            return await future

                        now = time.monotonic()
                        stall_sec = now - snapshot.last_progress_at
                        if stall_sec >= self._scan_timeout_sec:
                            self._logger.warning(
                                "[scan timeout] path=%s last_progress_sec_ago=%.1f "
                                "scan stalled and will be skipped",
                                path,
                                stall_sec,
                            )
                            return [], []
            finally:
                async with in_progress_lock:
                    in_progress_scans.pop(path, None)

        async def _stall_monitor() -> None:
            try:
                while True:
                    await asyncio.sleep(self._SCAN_STALL_LOG_INTERVAL_SEC)
                    now = time.monotonic()
                    async with in_progress_lock:
                        stalled = [
                            (
                                path,
                                now - progress.started_at,
                                now - progress.last_progress_at,
                            )
                            for path, progress in in_progress_scans.items()
                            if now - progress.started_at
                            >= self._SCAN_STALL_LOG_INTERVAL_SEC
                        ]

                    if stalled:
                        stalled.sort(key=lambda x: x[2], reverse=True)
                        self._logger.warning(
                            "[scan monitor] slow directories currently being scanned:"
                        )
                        for path, elapsed, stall in stalled[:10]:
                            self._logger.warning(
                                "  - %s (elapsed=%.1f sec, stalled=%.1f sec)",
                                path,
                                elapsed,
                                stall,
                            )
            except asyncio.CancelledError:
                return

        async def _walk(path: pathlib.Path) -> None:
            async with semaphore:
                dirs, hits = await _scan_with_timeout(path)
                files.extend(hits)
                progress_bar.update(len(hits))

            if dirs:
                await asyncio.gather(*(_walk(d) for d in dirs))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._scan_parallelism
        ) as pool:
            monitor_task = asyncio.create_task(_stall_monitor())
            try:
                await _walk(root)
            finally:
                monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor_task
                progress_bar.close()

        return sorted(files, key=lambda p: str(p).lower())

    @cache
    def load_all_image_item(self) -> List[ImageItem]:
        """
        画像ルート以下を走査し、キャッシュされた ImageItem 一覧を返す。

        create_index.py 由来の改善点:
        - Pillow の registered_extensions() を使って拡張子を決定
        - 64 並列を上限にディレクトリ単位で非同期走査
        - 進捗停滞時だけ timeout 扱いでスキップ
        - 遅いディレクトリを monitor ログに出す
        """
        extensions = ['.avif', '.avifs', '.blp', '.bmp', '.dib', '.bufr', '.cur', '.pcx', '.dcx', '.dds', '.ps', '.eps', '.fit', '.fits', '.fli', '.flc', '.ftc', '.ftu', '.gbr', '.gif', '.grib', '.h5', '.hdf', '.png', '.apng', '.jp2', '.j2k', '.jpc', '.jpf', '.jpx', '.j2c', '.icns', '.ico', '.im', '.iim', '.jfif', '.jpe', '.jpg', '.jpeg', '.mpg', '.mpeg', '.tif', '.tiff', '.mpo', '.msp', '.palm', '.pcd', '.pxr', '.pbm', '.pgm', '.ppm', '.pnm', '.pfm', '.psd', '.qoi', '.bw', '.rgb', '.rgba', '.sgi', '.ras', '.tga', '.icb', '.vda', '.vst', '.webp', '.wmf', '.emf', '.xbm', '.xpm']
        extensions = set(e.lower() if e.startswith('.') else f'.{e.lower()}' for e in extensions)
        files = self._run_coro_sync(self._collect_image_relative_paths(extensions))

        def create_image_item(relative_file: pathlib.Path):
            relative_file_str = str(relative_file)
            image_id = ImageId(
                hashlib.sha256(relative_file_str.encode("utf-8")).hexdigest()
            )
            image_item = ImageItem(
                image_id,
                ImageName(relative_file_str),
                ImageTags(relative_file_str),
            )
            return image_id, relative_file, image_item

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._IMAGE_ITEM_WORKERS
        ) as executor:
            results = list(
                executor.map(
                    create_image_item,
                    tqdm.tqdm(files, desc="ImageItemを作成中"),
                )
            )

        self._id_to_path.update(
            {image_id: relative_path for image_id, relative_path, _ in results}
        )

        items = [item for _, _, item in results]
        items.sort(key=lambda item: item.display_name.name)
        return items

    @cache
    def load_all_model_item(self) -> list[ModelItem]:
        """open_clip の事前学習モデル一覧を取得し、キャッシュ済みのModelItemに変換する。

        - ``open_clip.list_pretrained()`` が返す (モデル名, データセット) のペアを ``ModelItem`` にマッピングする。
        - 結果は ``functools.cache`` によりメモ化され、同一プロセス内で再計算を避ける。
        - 戻り値は ``ModelItem`` オブジェクトのリスト。
        """

        items: list[ModelItem] = [
            ModelItem(
                ModelId(model_name, dataset), ModelName(f"{model_name}-{dataset}")
            )
            for (model_name, dataset) in open_clip.list_pretrained()
        ]
        return items

    @cache
    def load_image(self, image_id: ImageId) -> Image:
        """フルサイズ画像を読み込み、MIME 推定結果とともに返す（キャッシュ対象）。

        - ``load_all_image_item`` で構築された ``_id_to_path`` を用いて相対パスを引き当てる。
        - 対応するファイルをバイナリとして読み込み、 ``mimetypes.guess_type`` でContent-Typeを推測する。
        - 結果は ``functools.cache`` によりメモ化され、同一IDの再読み込みを防ぐ。
        - 戻り値は ``Image`` ドメインオブジェクト（バイナリ本体とcontent_typeを保持）。
        """
        relative_path = self._id_to_path.get(image_id)

        if relative_path is None:
            self._logger.warning("指定されたImageIdが存在しません: %s", image_id)
            return None

        path: pathlib.Path = self._image_dir_path / relative_path
        binary: bytes = path.read_bytes()

        content_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'

        return Image(binary, content_type)

    @cache
    def load_small_image(self, image_id: ImageId) -> Image:
        """縮小サムネイルを生成して返す（400px以下、キャッシュ対象）。

        - ``_id_to_path`` から元画像パスを取得し、Pillow で開いて長辺が400pxになるよう ``thumbnail`` で縮小する。
        - 元のフォーマットを維持しつつバイナリへ保存する。形式が判別できない場合は PNG を使用。
        - ``mimetypes.guess_type`` による MIME 推定結果を ``Image`` に格納する。
        - 生成済みのサムネイルは ``functools.cache`` でメモ化され、再生成を回避する。
        """
        relative_path = self._id_to_path.get(image_id)

        if relative_path is None:
            self._logger.warning("指定されたImageIdが存在しません: %s", image_id)
            return None

        path: pathlib.Path = self._image_dir_path / relative_path

        # open image with PIL and resize to long side 400 px
        with PILImage.open(path) as img:
            img.thumbnail((400, 400))
            buffer = io.BytesIO()
            format = img.format if img.format else 'PNG'
            img.save(buffer, format=format)
            binary = buffer.getvalue()

        content_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'

        return Image(binary, content_type)

    def create_zip_from_images(self, images_with_names: list[tuple[Image, ImageName]]):
        """画像バイナリのリストからZIPを生成し、メモリ上のバッファを返す。

        - 引数は ``(Image, ImageName)`` のタプルのリストで、content_typeから拡張子を推定しつつ元の名前で格納する。
        - 圧縮形式は ``zipfile.ZIP_DEFLATED``、出力は ``io.BytesIO`` 上に書き込まれ ``seek(0)`` 済みで返却される。
        - 戻り値は ``BytesIO`` バッファで、呼び出し側がHTTPレスポンス等へ直接書き出せる。
        """
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for image, image_name in images_with_names:
                extension = image.content_type.split('/')[-1]
                filename = f"{image_name.name}"
                zip_file.writestr(filename, image.binary)
        zip_buffer.seek(0)
        return zip_buffer

    @cache
    def get_image_name(self, image_id: ImageId) -> ImageName:
        """画像IDに対応するファイル名だけを返す（キャッシュ対象）。

        - ``load_all_image_item`` が構築した ``_id_to_path`` を参照し、相対パスからファイル名を抽出する。
        - 見つからない場合は ``ValueError`` を送出し、成功時は ``ImageName`` ドメインオブジェクトを返す。
        - 関数結果は ``functools.cache`` によりメモ化される。
        """
        relative_path = self._id_to_path.get(image_id)
        if relative_path is None:
            raise ValueError(f"指定されたImageIdが存在しません: {image_id}")

        # パスからファイル名だけを取得
        image_name = pathlib.Path(relative_path).name

        return ImageName(name=image_name)

    def set_image_paths(self, image_paths: dict[ImageId, pathlib.Path]) -> None:
        """ImageIdから相対パスを引くためのマップを設定する。"""
        self._id_to_path = dict(image_paths)
