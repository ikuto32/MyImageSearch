"""画像ディレクトリを並列に走査して、画像ファイルの相対パスを集める。"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
from dataclasses import dataclass
import os
import pathlib
import threading
import time
from typing import NamedTuple, Sequence

from PIL import Image
import tqdm

# SMB のレイテンシを考慮した、同時に走査するディレクトリ数の上限。
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


class ImageScan(NamedTuple):
    # 指定拡張子のファイル。画像ルートからの相対パスで、整列済み。
    paths: list[str]
    # 中を調べられなかったディレクトリなど。paths と同じく画像ルートからの相対パスで、ルート自体は ""。
    # その下の画像は paths に含まれない。
    unreadable: list[str]


def scan_image_dir(
    dir_path: str | os.PathLike,
    exts: Sequence[str],
    *,
    scan_timeout_sec: float = _SCAN_TIMEOUT_SEC,
) -> ImageScan:
    """dir_path 以下を走査して、指定拡張子のファイルを集める。

    読めないディレクトリや、scan_timeout_sec のあいだ進捗のないディレクトリは飛ばして、
    残りの走査を続ける。飛ばした場所は結果の unreadable に入る。
    """
    return asyncio.run(
        _collect_images(pathlib.Path(dir_path), exts, scan_timeout_sec=scan_timeout_sec)
    )


async def _collect_images(
    root: pathlib.Path,
    exts: Sequence[str],
    *,
    scan_timeout_sec: float,
) -> ImageScan:
    loop = asyncio.get_running_loop()

    # .xyz と xyz の両方を許容
    extensions = tuple({("." + e if not e.startswith(".") else e).lower() for e in exts})

    files: list[str] = []
    unreadable: list[str] = []
    bar = tqdm.tqdm(unit="file", dynamic_ncols=True)

    in_progress_scans: dict[pathlib.Path, _ScanProgress] = {}
    in_progress_lock = asyncio.Lock()
    thread_progress_lock = threading.Lock()
    semaphore = asyncio.Semaphore(_PARALLEL)

    def mark_progress(path: pathlib.Path) -> None:
        now = time.monotonic()
        with thread_progress_lock:
            progress = in_progress_scans.get(path)
            if progress is not None:
                progress.last_progress_at = now

    def last_progress(path: pathlib.Path) -> float | None:
        with thread_progress_lock:
            progress = in_progress_scans.get(path)
            return None if progress is None else progress.last_progress_at

    def relative_location(path: pathlib.Path) -> str | None:
        """画像ルートからの相対パス。ルート自体は ""。予約名などで求められなければNone。"""
        try:
            relative = os.path.relpath(path, root)
        except ValueError:
            return None
        return "" if relative == os.curdir else relative

    def scan_directory(path: pathlib.Path) -> tuple[list[pathlib.Path], list[str], list[str]]:
        """ワーカースレッドで1ディレクトリを走査し、(サブディレクトリ, 画像の相対パス, 読めなかった場所)を返す。"""
        dirs, hits, skipped = [], [], []
        # ファイルごとにrelpathを呼ばないよう、ディレクトリの相対パスを先に求める。
        relative_dir = relative_location(path)
        if relative_dir is None:
            # 相対パスを求められないディレクトリは対象にしない。
            return dirs, hits, skipped
        prefix = relative_dir and relative_dir + os.sep
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    mark_progress(path)
                    # SMB では d_type がないことがあるので try/except
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(pathlib.Path(entry.path))
                        elif entry.name.lower().endswith(extensions):
                            hits.append(prefix + entry.name)
                    except OSError:
                        # 種類を調べられない項目は、中をたどれなかったディレクトリかもしれない。
                        skipped.append(prefix + entry.name)
        except OSError:
            skipped.append(relative_dir)
        return dirs, hits, skipped

    async def scan_with_timeout(path: pathlib.Path) -> tuple[list[pathlib.Path], list[str], list[str]]:
        start = time.monotonic()
        async with in_progress_lock:
            in_progress_scans[path] = _ScanProgress(started_at=start, last_progress_at=start)

        future = loop.run_in_executor(pool, scan_directory, path)
        try:
            while True:
                try:
                    return await asyncio.wait_for(asyncio.shield(future), timeout=scan_timeout_sec)
                except TimeoutError:
                    progressed_at = last_progress(path)
                    if progressed_at is None:
                        return await future
                    stall_sec = time.monotonic() - progressed_at
                    if stall_sec >= scan_timeout_sec:
                        print(
                            f"[scan timeout] path={path} "
                            f"last_progress_sec_ago={stall_sec:.1f} "
                            "scan stalled and will be skipped"
                        )
                        relative_dir = relative_location(path)
                        return [], [], [] if relative_dir is None else [relative_dir]
        finally:
            async with in_progress_lock:
                in_progress_scans.pop(path, None)

    async def stall_monitor() -> None:
        while True:
            await asyncio.sleep(_SCAN_STALL_LOG_INTERVAL_SEC)
            now = time.monotonic()
            async with in_progress_lock:
                stalled = [
                    (path, now - progress.started_at, now - progress.last_progress_at)
                    for path, progress in in_progress_scans.items()
                    if now - progress.started_at >= _SCAN_STALL_LOG_INTERVAL_SEC
                ]
            if stalled:
                stalled.sort(key=lambda x: x[2], reverse=True)
                print("[scan monitor] slow directories currently being scanned:")
                for path, elapsed, stall in stalled[:10]:
                    print(f"  - {path} (elapsed={elapsed:.1f} sec, stalled={stall:.1f} sec)")

    async def walk(path: pathlib.Path) -> None:
        async with semaphore:
            dirs, hits, skipped = await scan_with_timeout(path)
            files.extend(hits)
            unreadable.extend(skipped)
            bar.update(len(hits))
        # サブディレクトリを並列にたどる
        await asyncio.gather(*(walk(d) for d in dirs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL) as pool:
        monitor_task = asyncio.create_task(stall_monitor())
        try:
            await walk(root)
        finally:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task
            bar.close()

    # 末尾が「.」のディレクトリなど、別名で同じ場所を指すものがあっても1つのパスは1件にする。
    return ImageScan(sorted(set(files)), sorted(set(unreadable)))
