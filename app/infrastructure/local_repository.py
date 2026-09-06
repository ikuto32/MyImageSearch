import asyncio
import contextlib
import errno
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
import hashlib
import io
import itertools
import logging
import os
import pathlib
import mimetypes
import shutil
import threading
import time
from typing import List, Sequence
import concurrent.futures
import zipfile
import tempfile

from app.infrastructure.model_metadata import (
    default_model_dir_name,
    has_search_index,
    read_model_metadata,
)

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
from app.domain.errors import ResourceLimitError

from PIL import Image as PILImage, ImageOps, UnidentifiedImageError


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
    _THUMBNAIL_CACHE_BYTES = 64 * 1024 * 1024
    MAX_IMAGE_BYTES = 64 * 1024 * 1024
    MAX_IMAGE_PIXELS = 64_000_000
    MAX_ZIP_BYTES = 64 * 1024 * 1024 * 1024
    ZIP_MEMORY_BYTES = 8 * 1024 * 1024
    ZIP_DISK_RESERVE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        image_dir_path: pathlib.Path,
        meta_dir_path: pathlib.Path | None = None,
        *,
        scan_timeout_sec: float = _SCAN_TIMEOUT_SEC,
        scan_parallelism: int = _PARALLEL,
        thumbnail_cache_bytes: int = _THUMBNAIL_CACHE_BYTES,
    ) -> None:
        if thumbnail_cache_bytes < 0:
            raise ValueError("thumbnail_cache_bytes must be non-negative")
        self._logger = logging.getLogger(__name__)
        self._image_dir_path: pathlib.Path = image_dir_path
        self._resolved_image_dir = image_dir_path.resolve()
        self._meta_dir_path: pathlib.Path = meta_dir_path or pathlib.Path('./clip_meta')
        self._id_to_path: dict[ImageId, pathlib.Path] = {}
        self._scan_timeout_sec = scan_timeout_sec
        self._scan_parallelism = scan_parallelism
        self._thumbnail_cache_limit = thumbnail_cache_bytes
        self._thumbnail_cache: OrderedDict[ImageId, Image] = OrderedDict()
        self._thumbnail_cache_size = 0
        self._thumbnail_cache_lock = threading.Lock()
        self._image_paths_generation = 0

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
        """clip_meta に存在する検索モデル一覧を返す。

        meta_dir 配下に ``metafiles.index`` と ``sqlite_image_meta.db`` を持つ
        インデックスディレクトリだけを検出し、UI で選択可能な
        ModelItem として返す。``model_meta.json`` がある場合は、Qwen などの
        repo ID を安全なディレクトリ名へ変換したインデックスから元の
        ModelId と表示名を復元する。
        """

        items_by_key: dict[tuple[str, str], ModelItem] = {}

        for index_dir in self._iter_search_index_dirs():
            item = self._model_item_from_index_dir(index_dir)
            items_by_key[(item.id.model_name, item.id.pretrained)] = item

        return sorted(
            items_by_key.values(),
            key=lambda item: item.display_name.name.lower(),
        )

    def _iter_search_index_dirs(self) -> list[pathlib.Path]:
        if not self._meta_dir_path.is_dir():
            return []
        return sorted(
            (path for path in self._meta_dir_path.iterdir() if has_search_index(path)),
            key=lambda path: path.name.lower(),
        )

    def _model_item_from_index_dir(self, index_dir: pathlib.Path) -> ModelItem:
        metadata = read_model_metadata(index_dir)
        if metadata is not None:
            return metadata.to_model_item()

        model_name, separator, pretrained = index_dir.name.rpartition("-")
        if not separator:
            model_name = index_dir.name
            pretrained = ""
        return ModelItem(
            ModelId(model_name, pretrained),
            ModelName(default_model_dir_name(ModelId(model_name, pretrained))),
        )

    def load_image(self, image_id: ImageId) -> Image:
        """フルサイズ画像を読み込み、MIME 推定結果とともに返す。

        - ``load_all_image_item`` で構築された ``_id_to_path`` を用いて相対パスを引き当てる。
        - 対応するファイルをバイナリとして読み込み、 ``mimetypes.guess_type`` でContent-Typeを推測する。
        - 大きな画像の閲覧やZIP出力でメモリを保持し続けないよう、キャッシュしない。
        - 戻り値は ``Image`` ドメインオブジェクト（バイナリ本体とcontent_typeを保持）。
        """
        path = self._image_path(image_id)
        try:
            with path.open('rb') as source:
                binary = source.read(self.MAX_IMAGE_BYTES + 1)
        except OSError as error:
            raise ValueError(f"画像ファイルを開けません: {image_id.id}") from error
        if len(binary) > self.MAX_IMAGE_BYTES:
            raise ResourceLimitError("元画像は64MiB以内で指定してください。")

        content_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'

        return Image(binary, content_type)

    def _image_path(self, image_id: ImageId) -> pathlib.Path:
        """Resolve only image files contained by the configured image root."""
        relative_path = self._id_to_path.get(image_id)
        if relative_path is None:
            raise ValueError(f"画像IDが見つかりません: {image_id.id}")
        # Reject drive/UNC/absolute paths even if they happen to point inside root.
        relative_path = pathlib.Path(relative_path)
        if relative_path.is_absolute() or relative_path.drive:
            raise ValueError("画像パスは画像ディレクトリ内の相対パスである必要があります。")
        try:
            path = (self._resolved_image_dir / relative_path).resolve()
            path.relative_to(self._resolved_image_dir)
            if path.suffix.lower() not in PILImage.registered_extensions():
                raise ValueError("対応していない画像形式です。")
            if path.stat().st_size > self.MAX_IMAGE_BYTES:
                raise ResourceLimitError("元画像は64MiB以内で指定してください。")
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"画像ファイルを開けません: {image_id.id}") from error
        return path

    def load_small_image(self, image_id: ImageId) -> Image:
        """縮小サムネイルを生成して返す（400px以下、容量制限付きLRU）。

        - ``_id_to_path`` から元画像パスを取得し、Pillow で開いて長辺が400pxになるよう ``thumbnail`` で縮小する。
        - 元のフォーマットを維持しつつバイナリへ保存する。形式が判別できない場合は PNG を使用。
        - ``mimetypes.guess_type`` による MIME 推定結果を ``Image`` に格納する。
        - 生成済みバイナリは合計64MiB（既定）まで保持し、古い利用順で破棄する。
        - 上限より大きい画像はキャッシュせず返す。上限0でキャッシュを無効化できる。
        """
        with self._thumbnail_cache_lock:
            cached_image = self._thumbnail_cache.get(image_id)
            if cached_image is not None:
                self._thumbnail_cache.move_to_end(image_id)
                return cached_image
            relative_path = self._id_to_path.get(image_id)
            generation = self._image_paths_generation

        if relative_path is None:
            self._logger.warning("指定されたImageIdが存在しません: %s", image_id)
            raise ValueError(f"指定されたImageIdが存在しません: {image_id}")

        path = self._image_path(image_id)

        try:
            with PILImage.open(path) as img:
                if img.width * img.height > self.MAX_IMAGE_PIXELS:
                    raise ResourceLimitError("サムネイルにできる画像は6400万画素以内です。")
                format = img.format or 'PNG'
                img.thumbnail((400, 400))
                oriented = ImageOps.exif_transpose(img)
                buffer = io.BytesIO()
                oriented.save(buffer, format=format)
                binary = buffer.getvalue()
        except (OSError, UnidentifiedImageError, PILImage.DecompressionBombError) as error:
            raise ValueError(f"画像ファイルを開けません: {image_id.id}") from error

        content_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'

        image = Image(binary, content_type)
        size = len(binary)
        if self._thumbnail_cache_limit and size <= self._thumbnail_cache_limit:
            with self._thumbnail_cache_lock:
                # パス一覧の差替え中に生成した古い画像はキャッシュへ戻さない。
                if generation == self._image_paths_generation:
                    previous = self._thumbnail_cache.pop(image_id, None)
                    if previous is not None:
                        self._thumbnail_cache_size -= len(previous.binary)
                    while self._thumbnail_cache_size + size > self._thumbnail_cache_limit:
                        _, evicted = self._thumbnail_cache.popitem(last=False)
                        self._thumbnail_cache_size -= len(evicted.binary)
                    self._thumbnail_cache[image_id] = image
                    self._thumbnail_cache_size += size
        return image

    def create_zip_from_images(self, images_with_names):
        """(Image, ImageName) の反復可能オブジェクトから容量制限付きZIPを作る。

        出力は8MiBを超えるとディスクへ退避し、seek(0)済みで返す。
        呼び出し側は応答終了時にバッファをcloseする。
        """
        spool_directory = tempfile.gettempdir()
        zip_buffer = tempfile.SpooledTemporaryFile(max_size=self.ZIP_MEMORY_BYTES, mode="w+b", dir=spool_directory)
        total_bytes = 0
        used_names = set()
        try:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for image, image_name in images_with_names:
                    total_bytes += len(image.binary)
                    if total_bytes > self.MAX_ZIP_BYTES:
                        raise ResourceLimitError("画像の合計が64GiBを超えています。件数を減らしてダウンロードしてください。")
                    # ZIP's deflate output can be a little larger than its input.
                    # Roll over before writing an image, rather than temporarily
                    # retaining a large compressed member in the memory spool.
                    write_bound = len(image.binary) + len(image.binary) // 1000 + 65536
                    if zip_buffer._rolled or zip_buffer.tell() + write_bound > self.ZIP_MEMORY_BYTES:
                        buffered_bytes = 0 if zip_buffer._rolled else zip_buffer.tell()
                        required = buffered_bytes + write_bound + self.ZIP_DISK_RESERVE_BYTES
                        if shutil.disk_usage(spool_directory).free < required:
                            raise ResourceLimitError("ZIPを作成する一時ディスクの空き容量が不足しています。空きを確保するか保存枚数を減らしてください。")
                        zip_buffer.rollover()
                    # Preserve distinct images with the same basename and never
                    # emit extraction paths controlled by stored metadata.
                    base = pathlib.PureWindowsPath(image_name.name).name or 'image'
                    filename = base
                    suffix = 2
                    while filename.casefold() in used_names:
                        stem = pathlib.PurePath(base).stem
                        extension = pathlib.PurePath(base).suffix
                        filename = f"{stem} ({suffix}){extension}"
                        suffix += 1
                    used_names.add(filename.casefold())
                    zip_file.writestr(filename, image.binary)
            zip_buffer.seek(0)
            return zip_buffer
        except OSError as error:
            zip_buffer.close()
            if error.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)) or getattr(error, "winerror", None) in (39, 112):
                raise ResourceLimitError("ZIP作成中にディスクの空き容量が不足しました。空きを確保するか保存枚数を減らしてください。") from error
            raise
        except BaseException:
            zip_buffer.close()
            raise

    def get_image_name(self, image_id: ImageId) -> ImageName:
        """画像IDに対応するファイル名だけを返す。

        - ``load_all_image_item`` が構築した ``_id_to_path`` を参照し、相対パスからファイル名を抽出する。
        - 見つからない場合は ``ValueError`` を送出し、成功時は ``ImageName`` ドメインオブジェクトを返す。
        """
        relative_path = self._id_to_path.get(image_id)
        if relative_path is None:
            raise ValueError(f"指定されたImageIdが存在しません: {image_id}")

        # パスからファイル名だけを取得
        image_name = pathlib.Path(relative_path).name

        return ImageName(name=image_name)

    def register_image_items(self, items: list[ImageItem]) -> None:
        """Merge displayed search results without rescanning the image root.

        Files are still resolved and checked by _image_path when read. Reject
        obvious external paths here without adding a filesystem stat per result.
        """
        with self._thumbnail_cache_lock:
            changed_existing_path = False
            for item in items:
                relative_path = pathlib.Path(item.display_name.name)
                if relative_path.is_absolute() or relative_path.drive or ".." in relative_path.parts:
                    continue
                previous_path = self._id_to_path.get(item.id)
                if previous_path == relative_path:
                    continue
                self._id_to_path[item.id] = relative_path
                if previous_path is not None:
                    changed_existing_path = True
                    old_thumbnail = self._thumbnail_cache.pop(item.id, None)
                    if old_thumbnail is not None:
                        self._thumbnail_cache_size -= len(old_thumbnail.binary)
            if changed_existing_path:
                self._image_paths_generation += 1

    def set_image_paths(self, image_paths: dict[ImageId, pathlib.Path]) -> None:
        """ImageIdから相対パスを引くためのマップを設定する。"""
        with self._thumbnail_cache_lock:
            self._id_to_path = dict(image_paths)
            self._image_paths_generation += 1
            self._thumbnail_cache.clear()
            self._thumbnail_cache_size = 0
