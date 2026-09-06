"""Short-lived prepared downloads, held on disk and consumed once by the browser."""
from collections import OrderedDict
from dataclasses import dataclass
import secrets
import threading
import time


class DownloadBusyError(Exception):
    pass


@dataclass
class PreparedDownload:
    stream: object
    expires_at: float
    timer: threading.Timer


class DownloadStore:
    def __init__(self, ttl_seconds=300, max_pending=2):
        self.ttl_seconds = ttl_seconds
        self._slots = threading.BoundedSemaphore(max_pending)
        self._lock = threading.Lock()
        self._entries = OrderedDict()

    def prepare(self, build):
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
            return token, stream, size
        except BaseException:
            if stream is not None:
                stream.close()
            self._slots.release()
            raise

    def expire(self, token):
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is not None:
            entry.timer.cancel()
            entry.stream.close()
            self._slots.release()

    def consume(self, token):
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
    def __init__(self, stream, release):
        self._stream = stream
        self._release = release
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._stream.close()
            finally:
                self._release()
