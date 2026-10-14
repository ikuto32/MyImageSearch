from collections.abc import Callable, Hashable
import threading
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class OnceCache(Generic[K, V]):
    """キーごとに値を一度だけ構築して保持する。

    同じキーへの並列な初回要求は1回の構築を待ち合わせ、別のキーは互いに待たない。
    構築の失敗は保持せず、次の要求で再試行する。
    """

    def __init__(self) -> None:
        self._values: dict[K, V] = {}
        self._locks: dict[K, threading.Lock] = {}
        self._guard = threading.Lock()

    def __contains__(self, key: K) -> bool:
        return key in self._values

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: K, build: Callable[[], V]) -> V:
        with self._guard:
            if key in self._values:
                return self._values[key]
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            with self._guard:
                if key in self._values:
                    return self._values[key]
            value = build()
            with self._guard:
                self._values[key] = value
            return value
