"""Short-lived prepared downloads, held on disk and consumed once by the browser."""
from collections.abc import Callable
from dataclasses import dataclass
import secrets
import threading
import time
from typing import BinaryIO


class DownloadBusyError(Exception):
    """準備中・受け取り待ちのダウンロードが上限に達している。"""


@dataclass
class PreparedDownload:
    stream: BinaryIO
    expires_at: float
    timer: threading.Timer


class DownloadStore:
    def __init__(self, ttl_seconds: float = 300, max_pending: int = 2) -> None:
        self.ttl_seconds = ttl_seconds
        self._slots = threading.BoundedSemaphore(max_pending)
        self._lock = threading.Lock()
        self._entries: dict[str, PreparedDownload] = {}

    def prepare(self, build: Callable[[], BinaryIO]) -> tuple[str, int]:
        """空きがあればbuildでストリームを作り、受け取り用のトークンとバイト数を返す。"""
        if not self._slots.acquire(blocking=False):
            raise DownloadBusyError()
        stream = None
        try:
            stream = build()
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(0)
            token = secrets.token_urlsafe(32)
            timer = threading.Timer(self.ttl_seconds, self.expire, args=(token,))
            timer.daemon = True
            with self._lock:
                self._entries[token] = PreparedDownload(stream, time.monotonic() + self.ttl_seconds, timer)
            timer.start()
            return token, size
        except BaseException:
            if stream is not None:
                stream.close()
            self._slots.release()
            raise

    def expire(self, token: str) -> None:
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is not None:
            entry.timer.cancel()
            entry.stream.close()
            self._slots.release()

    def expire_all(self) -> None:
        with self._lock:
            tokens = list(self._entries)
        for token in tokens:
            self.expire(token)

    def consume(self, token: str) -> "ReleasingStream | None":
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is None:
            return None
        entry.timer.cancel()
        if time.monotonic() >= entry.expires_at:
            entry.stream.close()
            self._slots.release()
            return None
        # Keep the slot occupied until the response stream closes.
        return ReleasingStream(entry.stream, self._slots.release)


class ReleasingStream:
    """closeされたときに一度だけ、ストリームを閉じて枠を返す。

    接続が途中で切れると、サーバーは応答をcloseしないまま手放すことがある。
    その場合も枠が戻るよう、破棄されるときにもcloseする。
    """

    # __init__ の途中で破棄されても、__del__ と __getattr__ が安全に動くための既定値。
    _stream = None
    _closed = True

    def __init__(self, stream: BinaryIO, release: Callable[[], None]) -> None:
        self._stream = stream
        self._release = release
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._stream.close()
            finally:
                self._release()

    def __del__(self) -> None:
        self.close()
