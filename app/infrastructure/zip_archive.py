from collections.abc import Iterable
import errno
import pathlib
import shutil
import tempfile
from typing import BinaryIO
import zipfile

from app.domain.domain_object import Image
from app.domain.errors import ResourceLimitError

MAX_ZIP_BYTES = 64 * 1024 * 1024 * 1024
ZIP_MEMORY_BYTES = 8 * 1024 * 1024
ZIP_DISK_RESERVE_BYTES = 16 * 1024 * 1024

# Windowsのディスク容量不足(ERROR_HANDLE_DISK_FULL, ERROR_DISK_FULL)。
_WINDOWS_DISK_FULL_ERRORS = (39, 112)


def _unique_name(image: Image, used_names: set[str]) -> str:
    """Preserve distinct images with the same basename and never emit
    extraction paths controlled by stored metadata."""
    base = pathlib.PureWindowsPath(image.name).name or "image"
    name = base
    suffix = 2
    while name.casefold() in used_names:
        stem, extension = pathlib.PurePath(base).stem, pathlib.PurePath(base).suffix
        name = f"{stem} ({suffix}){extension}"
        suffix += 1
    used_names.add(name.casefold())
    return name


def create_zip(images: Iterable[Image]) -> BinaryIO:
    """画像から容量制限付きのZIPを作る。

    出力はZIP_MEMORY_BYTESを超えるとディスクへ退避し、seek(0)済みで返す。
    呼び出し側は応答終了時にストリームをcloseする。
    """
    spool_directory = tempfile.gettempdir()
    stream = tempfile.SpooledTemporaryFile(max_size=ZIP_MEMORY_BYTES, mode="w+b", dir=spool_directory)
    total_bytes = 0
    used_names: set[str] = set()
    try:
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for image in images:
                total_bytes += len(image.binary)
                if total_bytes > MAX_ZIP_BYTES:
                    raise ResourceLimitError("画像の合計が64GiBを超えています。件数を減らしてダウンロードしてください。")
                # ZIP's deflate output can be a little larger than its input.
                # Roll over before writing an image, rather than temporarily
                # retaining a large compressed member in the memory spool.
                write_bound = len(image.binary) + len(image.binary) // 1000 + 65536
                if stream._rolled or stream.tell() + write_bound > ZIP_MEMORY_BYTES:
                    buffered_bytes = 0 if stream._rolled else stream.tell()
                    required = buffered_bytes + write_bound + ZIP_DISK_RESERVE_BYTES
                    if shutil.disk_usage(spool_directory).free < required:
                        raise ResourceLimitError("ZIPを作成する一時ディスクの空き容量が不足しています。空きを確保するか保存枚数を減らしてください。")
                    stream.rollover()
                archive.writestr(_unique_name(image, used_names), image.binary)
        stream.seek(0)
        return stream
    except OSError as error:
        stream.close()
        disk_full = (
            error.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC))
            or getattr(error, "winerror", None) in _WINDOWS_DISK_FULL_ERRORS
        )
        if disk_full:
            raise ResourceLimitError("ZIP作成中にディスクの空き容量が不足しました。空きを確保するか保存枚数を減らしてください。") from error
        raise
    except BaseException:
        stream.close()
        raise
