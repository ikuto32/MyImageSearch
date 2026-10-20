from collections import OrderedDict
from contextlib import closing
import sqlite3
import threading

from app.domain.domain_object import ImageId, ImageItem, ModelId
from app.domain.rating import KNOWN_RATINGS, UNCLASSIFIED, RatingFilter
from app.infrastructure.index_files import image_meta_columns, open_readonly_database, select_list
from app.infrastructure.model_registry import ModelRegistry

_SQLITE_MAX_INTEGER = 2**63 - 1
# 絞り込みのない一覧では、これより離れた位置へはpath indexだけを使って移動する。
_MAX_UNFILTERED_SKIP = 1024
_MAX_PAGE_ANCHORS = 256

_RATING_CATEGORY_SQL = (
    "CASE WHEN rating IN ({}) THEN rating ELSE '{}' END"
    .format(",".join(f"'{rating}'" for rating in KNOWN_RATINGS), UNCLASSIFIED)
)
# 一覧には表示と絞り込みに使う列だけを読む。タグなどの詳細は画像ごとに読む。
_PAGE_COLUMNS = ("image_id", "image_path", "rating")
_DETAIL_COLUMNS = (
    "image_id", "image_path", "image_tags", "rating", "style_cluster", "aesthetic_quality", "image_blake3",
)


def _listed_item(row) -> ImageItem:
    identifier, path, rating = row
    return ImageItem(ImageId(str(identifier)), str(path), rating=rating or "")


def _detailed_item(row) -> ImageItem:
    identifier, path, tags, rating, style_cluster, quality, file_hash = row
    return ImageItem(
        id=ImageId(str(identifier)),
        name=str(path),
        tags=tags or "",
        rating=rating or "",
        style_cluster=style_cluster or "",
        aesthetic_quality=None if quality is None else float(quality),
        blake3=file_hash or "",
    )


class CatalogDatabase:
    """登録済みモデルの画像メタデータを、FAISSや埋め込みモデルを読み込まずに参照する。

    大規模な一覧を保持しないよう、件数と最近読んだページの境界だけを記憶する。
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._columns: dict[ModelId, frozenset[str]] = {}
        self._counts: dict[tuple[ModelId, RatingFilter], int] = {}
        self._histograms: dict[ModelId, dict[str, int]] = {}
        self._count_lock = threading.Lock()
        self._null_path_counts: dict[ModelId, int] = {}
        # (model, ratings) -> {先頭からの順位: (image_path, image_id)}
        self._anchors: dict[tuple[ModelId, RatingFilter], OrderedDict[int, tuple[str, str]]] = {}
        self._anchor_lock = threading.Lock()

    def _connect(self, model_id: ModelId) -> sqlite3.Connection:
        return open_readonly_database(self._registry.files(model_id).database)

    def _table_columns(self, model_id: ModelId, connection: sqlite3.Connection) -> frozenset[str]:
        columns = self._columns.get(model_id)
        if columns is None:
            columns = self._columns[model_id] = image_meta_columns(connection)
        return columns

    @staticmethod
    def _rating_condition(columns: frozenset[str], ratings: RatingFilter) -> tuple[list[str], list] | None:
        """絞り込みのWHERE条件とパラメータ。どの行にも一致しない場合はNone。"""
        if ratings is None:
            return [], []
        if "rating" not in columns:
            # ratingを持たない古いデータベースの画像は、すべて未分類として扱う。
            return ([], []) if UNCLASSIFIED in ratings else None
        if not ratings:
            return None
        placeholders = ",".join("?" for _ in ratings)
        return [f"{_RATING_CATEGORY_SQL} IN ({placeholders})"], list(ratings)

    # ===================================================================

    def count(self, model_id: ModelId, ratings: RatingFilter = None) -> int:
        key = (model_id, ratings)
        if key in self._counts:
            return self._counts[key]
        with self._count_lock:
            if key in self._counts:
                return self._counts[key]
            with closing(self._connect(model_id)) as connection:
                if ratings is None:
                    count = self._count_all(connection)
                elif not ratings:
                    count = 0
                else:
                    histogram = self._rating_histogram(model_id, connection)
                    count = sum(histogram.get(category, 0) for category in ratings)
            self._counts[key] = count
            return count

    @staticmethod
    def _count_all(connection: sqlite3.Connection) -> int:
        return int(connection.execute("SELECT COUNT(*) FROM image_meta").fetchone()[0])

    def _rating_histogram(self, model_id: ModelId, connection: sqlite3.Connection) -> dict[str, int]:
        """区分ごとの件数。呼び出し側が_count_lockを保持していること。

        The source database may have no rating index. Aggregate all categories
        once in SQL; never transfer the catalog to Python or repeat a full
        scan for each filter combination.
        """
        histogram = self._histograms.get(model_id)
        if histogram is None:
            if "rating" in self._table_columns(model_id, connection):
                rows = connection.execute(f"SELECT {_RATING_CATEGORY_SQL}, COUNT(*) FROM image_meta GROUP BY 1")
                histogram = {category: int(total) for category, total in rows}
            else:
                histogram = {UNCLASSIFIED: self._count_all(connection)}
            self._histograms[model_id] = histogram
            self._counts[(model_id, None)] = sum(histogram.values())
        return histogram

    # ===================================================================

    def page(self, model_id: ModelId, page: int, size: int, ratings: RatingFilter = None) -> list[ImageItem]:
        """image_path・image_id順の1ページ分を、必要な行だけ読んで返す。"""
        if type(page) is not int or page < 0 or type(size) is not int or size < 1:
            raise ValueError("page must be non-negative and size must be positive integers")
        offset = page * size
        # SQLite offsets are signed 64-bit; a later page cannot contain any row.
        if offset > _SQLITE_MAX_INTEGER:
            return []
        limit = min(size, _SQLITE_MAX_INTEGER)
        key = (model_id, ratings)
        with closing(self._connect(model_id)) as connection:
            columns = self._table_columns(model_id, connection)
            condition = self._rating_condition(columns, ratings)
            if condition is None:
                return []
            conditions, parameters = condition

            direction, skip = "ASC", offset
            total = self._counts.get(key)
            if total is not None:
                if offset >= total:
                    return []
                limit = min(limit, total - offset)
                from_end = total - offset - limit
                if from_end < offset and self._may_skip(ratings, from_end):
                    # Tail jumps should read from the end of the path index.
                    direction, skip = "DESC", from_end

            anchor = None
            with self._anchor_lock:
                anchors = list(self._anchors.get(key, {}).items())
            for rank, boundary in anchors:
                ahead = offset - rank
                if 0 <= ahead < skip and self._may_skip(ratings, ahead):
                    direction, skip, anchor = "ASC", ahead, boundary
                behind = rank - (offset + limit - 1)
                # SQLite row-value ranges omit NULL paths. Use the ordinary
                # reverse scan when NULL rows could be part of this range.
                if (0 <= behind < skip and self._may_skip(ratings, behind)
                        and not self._null_path_count(model_id, connection)):
                    direction, skip, anchor = "DESC", behind, boundary

            if anchor is not None:
                comparison = ">=" if direction == "ASC" else "<="
                conditions.append(f"(image_path, image_id) {comparison} (?, ?)")
                parameters.extend(anchor)
            elif ratings is None and direction == "ASC" and skip > _MAX_UNFILTERED_SKIP:
                # Seek using only the covering image_path index before reading
                # metadata. OFFSET on the full projection otherwise looks up
                # millions of discarded table rows. Preserve image_id tie order
                # by expanding the boundary path group and offsetting inside it.
                boundary = connection.execute(
                    "SELECT image_path FROM image_meta ORDER BY image_path LIMIT 1 OFFSET ?", (skip,)
                ).fetchone()
                if boundary is None:
                    return []
                if boundary[0] is not None:
                    before = connection.execute("""SELECT
                        (SELECT COUNT(*) FROM image_meta WHERE image_path < ?) +
                        (SELECT COUNT(*) FROM image_meta WHERE image_path IS NULL)
                    """, (boundary[0],)).fetchone()[0]
                    conditions.append("image_path >= ?")
                    parameters.append(boundary[0])
                    skip -= int(before)

            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            rows = connection.execute(f"""
                SELECT {select_list(columns, _PAGE_COLUMNS)}
                FROM image_meta{where} ORDER BY image_path {direction}, image_id {direction} LIMIT ? OFFSET ?
            """, (*parameters, limit, skip)).fetchall()
        if direction == "DESC":
            rows.reverse()
        if rows:
            self._remember_anchors(key, offset, rows)
        return [_listed_item(row) for row in rows]

    @staticmethod
    def _may_skip(ratings: RatingFilter, rows: int) -> bool:
        # 絞り込みがある場合はpath indexだけでは移動できないため、近い側から読むしかない。
        return ratings is not None or rows <= _MAX_UNFILTERED_SKIP

    def _null_path_count(self, model_id: ModelId, connection: sqlite3.Connection) -> int:
        if model_id not in self._null_path_counts:
            self._null_path_counts[model_id] = int(connection.execute(
                "SELECT COUNT(*) FROM image_meta WHERE image_path IS NULL"
            ).fetchone()[0])
        return self._null_path_counts[model_id]

    def _remember_anchors(self, key, offset: int, rows: list) -> None:
        with self._anchor_lock:
            saved = self._anchors.setdefault(key, OrderedDict())
            for rank, row in ((offset, rows[0]), (offset + len(rows) - 1, rows[-1])):
                if row[0] is not None and row[1] is not None:
                    saved[rank] = (row[1], row[0])
                    saved.move_to_end(rank)
            # Keep only boundaries of recently read pages, never the
            # full catalog; nearby scrolling then uses indexed seeks.
            while len(saved) > _MAX_PAGE_ANCHORS:
                saved.popitem(last=False)

    # ===================================================================

    def find(self, model_id: ModelId, image_id: ImageId) -> ImageItem | None:
        """主キーで1件を読む。タグや評価値を含む。"""
        with closing(self._connect(model_id)) as connection:
            fields = select_list(self._table_columns(model_id, connection), _DETAIL_COLUMNS)
            row = connection.execute(
                f"SELECT {fields} FROM image_meta WHERE image_id = ?", (image_id,)
            ).fetchone()
        return None if row is None else _detailed_item(row)

    def find_path(self, model_id: ModelId, image_id: ImageId) -> str | None:
        """画像IDに対応する、画像ルートからの相対パス。"""
        with closing(self._connect(model_id)) as connection:
            row = connection.execute(
                "SELECT image_path FROM image_meta WHERE image_id = ?", (image_id,)
            ).fetchone()
        return None if row is None or row[0] is None else str(row[0])
