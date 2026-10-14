from collections import OrderedDict
import io
import mimetypes
import pathlib
import threading

from PIL import Image as PILImage, ImageOps

from app.domain import limits
from app.domain.domain_object import Image
from app.domain.errors import ImageUnavailableError, ResourceLimitError

THUMBNAIL_SIZE = (400, 400)
# PNGとして保存できる画素形式。
_PNG_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"})
DEFAULT_THUMBNAIL_CACHE_BYTES = 64 * 1024 * 1024


class ImageStore:
    """画像ルート内のファイルだけを読み、縮小画像を容量上限つきで保持する。"""

    def __init__(
        self,
        image_dir: pathlib.Path,
        *,
        thumbnail_cache_bytes: int = DEFAULT_THUMBNAIL_CACHE_BYTES,
    ) -> None:
        if thumbnail_cache_bytes < 0:
            raise ValueError("thumbnail_cache_bytes must be non-negative")
        self._root = pathlib.Path(image_dir).resolve()
        self._thumbnail_limit = thumbnail_cache_bytes
        # 相対パス -> 縮小画像。利用の古い順に並ぶ。
        self._thumbnails: OrderedDict[str, Image] = OrderedDict()
        self._thumbnail_bytes = 0
        self._thumbnail_lock = threading.Lock()

    def _resolve(self, relative_path: str) -> pathlib.Path:
        """Resolve only image files contained by the configured image root."""
        relative = pathlib.Path(relative_path)
        # Reject drive/UNC/absolute paths even if they happen to point inside root.
        if relative.is_absolute() or relative.drive:
            raise ImageUnavailableError("画像パスは画像ディレクトリ内の相対パスである必要があります。")
        try:
            path = (self._root / relative).resolve()
            path.relative_to(self._root)
            if path.suffix.lower() not in PILImage.registered_extensions():
                raise ValueError("対応していない画像形式です。")
            size = path.stat().st_size
        except (OSError, RuntimeError, ValueError) as error:
            raise ImageUnavailableError(f"画像ファイルを開けません: {relative_path}") from error
        if size > limits.MAX_IMAGE_BYTES:
            raise ResourceLimitError("元画像は64MiB以内で指定してください。")
        return path

    @staticmethod
    def _content_type(path: pathlib.Path) -> str:
        return mimetypes.guess_type(path)[0] or "application/octet-stream"

    def load_original(self, relative_path: str) -> Image:
        """元のファイルをそのまま返す。大きな画像を保持し続けないよう、キャッシュしない。"""
        path = self._resolve(relative_path)
        try:
            with path.open("rb") as source:
                binary = source.read(limits.MAX_IMAGE_BYTES + 1)
        except OSError as error:
            raise ImageUnavailableError(f"画像ファイルを開けません: {relative_path}") from error
        if len(binary) > limits.MAX_IMAGE_BYTES:
            raise ResourceLimitError("元画像は64MiB以内で指定してください。")
        return Image(binary, self._content_type(path), relative_path)

    def load_thumbnail(self, relative_path: str) -> Image:
        """長辺400px以下の縮小画像を返す。元の形式を保ち、EXIFの向きを反映する。

        PSDなど、Pillowが読めても同じ形式では書けない画像はPNGにする。

        生成済みの画像は合計が上限に収まる範囲で保持し、利用の古い順に破棄する。
        上限より大きい画像は保持せずに返す。上限0でキャッシュを無効化できる。
        """
        with self._thumbnail_lock:
            cached = self._thumbnails.get(relative_path)
            if cached is not None:
                self._thumbnails.move_to_end(relative_path)
                return cached

        path = self._resolve(relative_path)
        try:
            with PILImage.open(path) as source:
                if source.width * source.height > limits.MAX_IMAGE_PIXELS:
                    raise ResourceLimitError("サムネイルにできる画像は6400万画素以内です。")
                content_type = self._content_type(path)
                image_format = source.format or "PNG"
                source.thumbnail(THUMBNAIL_SIZE)
                thumbnail = ImageOps.exif_transpose(source)
                buffer = io.BytesIO()
                try:
                    thumbnail.save(buffer, format=image_format)
                except (KeyError, OSError, ValueError):
                    # 書き込みに対応しない形式(KeyError)や、読んだ画素形式のままでは書けない形式。
                    content_type = "image/png"
                    if thumbnail.mode not in _PNG_MODES:
                        thumbnail = thumbnail.convert("RGB")
                        # 変換前の色空間(CMYKなど)のプロファイルは、変換後の画素に合わない。
                        thumbnail.info.pop("icc_profile", None)
                    buffer = io.BytesIO()
                    thumbnail.save(buffer, format="PNG")
        except PILImage.DecompressionBombError:
            raise ResourceLimitError("サムネイルにできる画像は6400万画素以内です。") from None
        except (OSError, ValueError, SyntaxError) as error:
            raise ImageUnavailableError(f"画像ファイルを開けません: {relative_path}") from error

        image = Image(buffer.getvalue(), content_type, relative_path)
        self._remember_thumbnail(relative_path, image)
        return image

    def _remember_thumbnail(self, relative_path: str, image: Image) -> None:
        size = len(image.binary)
        if not self._thumbnail_limit or size > self._thumbnail_limit:
            return
        with self._thumbnail_lock:
            # 並列に生成された同じ画像は、容量を二重に数えない。
            previous = self._thumbnails.pop(relative_path, None)
            if previous is not None:
                self._thumbnail_bytes -= len(previous.binary)
            while self._thumbnail_bytes + size > self._thumbnail_limit:
                _, evicted = self._thumbnails.popitem(last=False)
                self._thumbnail_bytes -= len(evicted.binary)
            self._thumbnails[relative_path] = image
            self._thumbnail_bytes += size
