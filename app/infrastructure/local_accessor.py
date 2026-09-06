from functools import cache
import hashlib
import json
import logging
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple

import pathlib

import numpy as np
import pandas as pd
import faiss
import open_clip
import torch

from app.domain.domain_object import (
    ImageId,
    ImageItem,
    ImageName,
    ImageTags,
    ModelId,
    Model,
    Tokenizer,
)
from app.application.accessor import Accessor
from app.domain.errors import SearchInputError
from app.application.embedding_backend import SearchEmbeddingBackend, to_float32_2d_array
from app.infrastructure.model_metadata import (
    has_search_index,
    normalize_model_id,
    read_model_metadata,
    INDEX_MANIFEST_SUFFIX,
    open_readonly_database,
    read_index_manifest,
)


class OpenClipEmbeddingBackend(SearchEmbeddingBackend):
    """既存のOpenCLIP検索エンコーダをSearchEmbeddingBackendとして扱う。"""

    def __init__(self, model_name: str, pretrained: str) -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            return to_float32_2d_array(self.model.encode_text(self.tokenizer([text])))

    def encode_image(self, image) -> np.ndarray:
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to("cpu")
            return to_float32_2d_array(self.model.encode_image(image_tensor))


class QwenEmbeddingBackend(SearchEmbeddingBackend):
    """Hugging FaceのQwen系埋め込みモデルを使う検索エンコーダ。"""

    def __init__(self, model_id: str) -> None:
        import transformers
        from transformers import AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
        self.model = transformers.Qwen3VLModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()

    def _move_inputs_to_device(self, inputs):
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, Mapping):
            return {
                key: self._move_inputs_to_device(value)
                for key, value in inputs.items()
            }
        if isinstance(inputs, tuple):
            return tuple(self._move_inputs_to_device(value) for value in inputs)
        if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, bytearray)):
            return [self._move_inputs_to_device(value) for value in inputs]
        return inputs

    def _structured_text_inputs(self, text: str):
        message = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        if hasattr(self.processor, "apply_chat_template"):
            try:
                prompt = self.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return self.processor(text=[prompt], padding=True, return_tensors="pt")
            except Exception:
                pass
        return self.processor(text=[text], padding=True, return_tensors="pt")

    def _structured_image_inputs(self, image):
        message = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        if hasattr(self.processor, "apply_chat_template"):
            try:
                prompt = self.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
            except Exception:
                pass
        return self.processor(images=image, return_tensors="pt")

    def _pool_outputs(self, outputs):
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(outputs, attr, None)
            if value is not None:
                return value
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        if hidden is None:
            raise RuntimeError("Qwen embedding model did not return embeddings.")
        return hidden.mean(dim=1)

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self._move_inputs_to_device(self._structured_text_inputs(text))
            outputs = self.model(**inputs, return_dict=True)
            return to_float32_2d_array(self._pool_outputs(outputs))

    def encode_image(self, image) -> np.ndarray:
        with torch.no_grad():
            inputs = self._move_inputs_to_device(self._structured_image_inputs(image))
            outputs = self.model(**inputs, return_dict=True)
            return to_float32_2d_array(self._pool_outputs(outputs))


def _is_qwen_model(model_id: ModelId) -> bool:
    return "qwen" in model_id.model_name.lower() or "qwen" in model_id.pretrained.lower()


def _hf_model_name(model_id: ModelId) -> str:
    if "/" in model_id.model_name or not model_id.pretrained:
        return model_id.model_name
    if "/" in model_id.pretrained:
        return model_id.pretrained
    return model_id.model_name


class LocalAccessor(Accessor):
    """ローカル上のファイルを対象としたAccessor"""

    def __init__(self, meta_dir_path) -> None:
        self._logger = logging.getLogger(__name__)
        self._meta_dir_path = pathlib.Path(meta_dir_path)
        self._id_to_path: Dict[ImageId, pathlib.Path] = {}
        self._search_indexes: dict[pathlib.Path, Any] = {}
        self._search_index_hashes: dict[pathlib.Path, str] = {}
        self._search_index_lock = threading.Lock()
        self._embedding_backends: dict[ModelId, SearchEmbeddingBackend] = {}
        self._embedding_backend_locks: dict[ModelId, threading.Lock] = {}
        self._embedding_backend_guard = threading.Lock()
        self._catalog_counts = {}
        self._catalog_rating_counts = {}
        self._catalog_count_lock = threading.RLock()
        self._catalog_page_anchors = {}
        self._catalog_anchor_lock = threading.Lock()
        self._catalog_null_path_counts = {}

    def _search_model_meta_dir(self, model_id: ModelId) -> pathlib.Path:
        """Resolve only locally registered indexes contained in the metadata root."""
        model_id = normalize_model_id(model_id)
        root = self._meta_dir_path.resolve()
        if root.is_dir():
            for index_dir in sorted(root.iterdir()):
                resolved_dir = index_dir.resolve()
                if not resolved_dir.is_relative_to(root):
                    continue
                if not has_search_index(index_dir):
                    continue
                artifact_names = (
                    "metafiles.index", "sqlite_image_meta.db", "model_meta.json",
                    "metafiles.index.mean.npy", "metafiles.index.meta.json",
                    "metafiles.index" + INDEX_MANIFEST_SUFFIX,
                )
                if any(not (index_dir / name).resolve().is_relative_to(root) for name in artifact_names):
                    continue
                metadata = read_model_metadata(index_dir)
                if metadata is not None:
                    registered_id = normalize_model_id(metadata.to_model_item().id)
                else:
                    model_name, separator, pretrained = index_dir.name.rpartition("-")
                    registered_id = ModelId(model_name, pretrained) if separator else ModelId(index_dir.name, "")
                if registered_id == model_id:
                    return resolved_dir
        raise SearchInputError(f"Search model is not registered in {root}: {model_id}")

    def load_image_feature(self, model_id: ModelId, image_id: ImageId) -> np.ndarray:
        """meta_dir配下のmodel-pretrained/image_id.npyから画像特徴量を読み込み、毎回np.loadで返す（キャッシュ無し）。"""
        return np.load(
            str(self._search_model_meta_dir(model_id) / f"{image_id}.npy")
        )

    @cache
    def load_model(self, model_id: ModelId) -> Model:
        """モデル名とpretrained設定でopen_clipモデル+変換器を構築しModelに包んで返す（結果は@cacheで共有）。"""

        model_name, pretrained = model_id.model_name, model_id.pretrained
        return Model(
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        )

    @cache
    def load_tokenizer(self, model_id: ModelId) -> Tokenizer:
        """model_id.model_nameに対応するopen_clipトークナイザを生成し、@cacheで同一インスタンスを再利用する。"""
        return Tokenizer(open_clip.get_tokenizer(model_id.model_name))


    def load_embedding_backend(self, model_id: ModelId) -> SearchEmbeddingBackend:
        """初回の並列リクエストでも同一モデルを一度だけロードする。"""

        model_id = normalize_model_id(model_id)
        with self._embedding_backend_guard:
            cached = self._embedding_backends.get(model_id)
            if cached is not None:
                return cached
        index_dir = self._search_model_meta_dir(model_id)
        metadata = read_model_metadata(index_dir)
        if metadata is not None:
            model_id = normalize_model_id(metadata.to_model_item().id)
        # Validate local registration before retaining a lock for this model ID.
        with self._embedding_backend_guard:
            model_lock = self._embedding_backend_locks.setdefault(model_id, threading.Lock())
        with model_lock:
            with self._embedding_backend_guard:
                cached = self._embedding_backends.get(model_id)
                if cached is not None:
                    return cached
            # Construction is outside the guard so different models can initialize
            # independently. Failures leave the cache empty for a later retry.
            if _is_qwen_model(model_id):
                backend = QwenEmbeddingBackend(_hf_model_name(model_id))
            else:
                backend = OpenClipEmbeddingBackend(model_id.model_name, model_id.pretrained)
            with self._embedding_backend_guard:
                self._embedding_backends[model_id] = backend
            return backend

    @cache
    def load_index_with_metadata(
        self, model_id: ModelId, aesthetic_model_name: str
    ) -> Tuple[Any, List[ImageItem]]:
        """FAISSを共有し、追加された行と正確に同じ順序のメタデータを返す。"""
        index_dir = self._search_model_meta_dir(model_id)
        index_path = index_dir / "metafiles.index"
        con = open_readonly_database(index_dir / "sqlite_image_meta.db")
        con.row_factory = sqlite3.Row
        try:
            manifest_path = pathlib.Path(str(index_path) + INDEX_MANIFEST_SUFFIX)
            has_manifest = manifest_path.is_file()
            if has_manifest:
                con.execute("ATTACH DATABASE ? AS manifest", (manifest_path.as_uri() + "?mode=ro",))
            con.execute("BEGIN")
            # The header and row mapping come from one attached SQLite snapshot,
            # even if a new manifest is published concurrently.
            manifest = read_index_manifest(index_path, connection=con, schema="manifest") if has_manifest else None
            self._check_required_manifest(index_path, manifest)
            index = self._load_search_index(index_path, manifest["index_sha256"] if manifest else None)
            if manifest is not None:
                if manifest["dimension"] != index.d or manifest["count"] != index.ntotal:
                    raise ValueError("Index manifest dimension/count does not match FAISS. Rebuild the index.")
                invalid_rows = con.execute("""
                    SELECT COUNT(*) FROM manifest.index_rows AS r
                    LEFT JOIN image_meta AS m ON m.image_id = r.image_id
                    WHERE m.image_id IS NULL OR m.image_path IS NOT r.image_path
                """).fetchone()[0]
                if invalid_rows:
                    raise ValueError("Index manifest references missing or changed image paths. Rebuild the index.")
                fields = self._metadata_columns(con, "m.")
                rows = con.execute(f"""
                    SELECT {fields}
                    FROM manifest.index_rows AS r JOIN image_meta AS m ON m.image_id = r.image_id
                    ORDER BY r.position
                """)
            else:
                # Legacy builders added only float32 vectors of the index dimension,
                # in SQLite image_path order. NULL/other-dimension rows were skipped.
                fields = self._metadata_columns(con)
                rows = con.execute(f"""
                    SELECT {fields}
                    FROM image_meta
                    WHERE typeof(meta) = 'blob' AND length(meta) = ?
                    ORDER BY image_path
                """, (int(index.d) * 4,))
            image_items = [self._image_item_from_row(row, aesthetic_model_name) for row in rows]
            if len(image_items) != index.ntotal:
                raise ValueError(
                    f"Search index contains {index.ntotal} vectors, but matching metadata contains "
                    f"{len(image_items)} images. Rebuild the index and metadata together."
                )
        finally:
            con.close()
        return index, image_items

    @staticmethod
    def _metadata_columns(con, prefix="") -> str:
        """Older databases predate optional rating/style/aesthetic columns."""
        available = {row[1] for row in con.execute("PRAGMA table_info(image_meta)")}
        if not {"image_id", "image_path"} <= available:
            raise ValueError("Image metadata must contain image_id and image_path columns.")
        columns = (
            "image_id", "image_path", "image_tags", "aesthetic_quality",
            "pony_aesthetic_quality", "style_cluster", "rating",
        )
        return ", ".join(f"{prefix}{name}" if name in available else f"NULL AS {name}" for name in columns)

    @staticmethod
    def _image_item_from_row(row, aesthetic_model_name="original") -> ImageItem:
        aesthetic_name = "aesthetic_quality" if aesthetic_model_name == "original" else "pony_aesthetic_quality"
        score = row[aesthetic_name]
        if score is not None and pd.isna(score):
            score = None
        return ImageItem(
            id=ImageId(str(row["image_id"])),
            display_name=ImageName(str(row["image_path"])),
            tags=ImageTags(row["image_tags"] or ""),
            aesthetic_quality=float(score) if score is not None else None,
            rating=row["rating"] or "",
            style_cluster=row["style_cluster"] or "",
        )

    def load_image_metadata(
        self, model_id: ModelId, image_ids: list[ImageId] | None = None
    ) -> list[ImageItem]:
        """Read metadata by primary key without loading FAISS or embedding models."""
        if image_ids == []:
            return []
        index_dir = self._search_model_meta_dir(model_id)
        con = open_readonly_database(index_dir / "sqlite_image_meta.db")
        con.row_factory = sqlite3.Row
        try:
            fields = self._metadata_columns(con)
            if image_ids is None:
                return [self._image_item_from_row(row) for row in con.execute(f"SELECT {fields} FROM image_meta")]
            requested_ids = list(dict.fromkeys(image_id.id for image_id in image_ids))
            items = []
            for start in range(0, len(requested_ids), 900):
                chunk = requested_ids[start:start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = con.execute(f"SELECT {fields} FROM image_meta WHERE image_id IN ({placeholders})", chunk)
                items.extend(self._image_item_from_row(row) for row in rows)
            return items
        finally:
            con.close()

    def load_download_image_items(
        self, model_id: ModelId, limit: int, ratings: list[str] | None = None,
    ) -> list[ImageItem]:
        """Select the first matching images in SQL without materializing the catalog."""
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise SearchInputError("Download limit must be from 1 to 1024")
        normalized_ratings = self._normalize_catalog_ratings(ratings)
        index_dir = self._search_model_meta_dir(model_id)
        if self._catalog_counts.get((normalize_model_id(model_id), normalized_ratings)) == 0:
            return []
        con = open_readonly_database(index_dir / "sqlite_image_meta.db")
        con.row_factory = sqlite3.Row
        try:
            fields = self._metadata_columns(con)
            where, parameters = self._catalog_rating_where(con, normalized_ratings)
            parameters.append(limit)
            rows = con.execute(
                f"SELECT {fields} FROM image_meta{where} ORDER BY image_path, image_id LIMIT ?",
                parameters,
            )
            return [self._image_item_from_row(row) for row in rows]
        finally:
            con.close()

    @staticmethod
    def _check_required_manifest(index_path, manifest):
        sidecar = pathlib.Path(str(index_path) + ".meta.json")
        if sidecar.is_file():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("index metadata must be an object")
            except (OSError, ValueError) as error:
                raise ValueError(f"Invalid index metadata: {sidecar}") from error
            if metadata.get("rows_manifest_version") is not None and manifest is None:
                raise ValueError("Index row manifest is missing. Rebuild the index.")

    def _load_search_index(self, index_path: pathlib.Path, expected_sha256: str | None = None):
        """同時検索でも同一ファイルの巨大なFAISS indexを一度だけ読み込む。"""
        index_path = index_path.resolve()
        with self._search_index_lock:
            if index_path not in self._search_indexes:
                # Hash the same file handle FAISS reads; avoid a second pass over a
                # potentially huge file and detect stale manifests after a rebuild.
                digest = hashlib.sha256()
                with index_path.open("rb") as stream:
                    def read_bytes(size):
                        data = stream.read(size)
                        digest.update(data)
                        return data
                    index = faiss.read_index(faiss.PyCallbackIOReader(read_bytes))
                    while read_bytes(1024 * 1024):
                        pass
                actual_hash = digest.hexdigest()
                if expected_sha256 is not None and actual_hash != expected_sha256:
                    raise ValueError("Index file and row manifest belong to different builds. Rebuild the index.")
                self._search_indexes[index_path] = index
                self._search_index_hashes[index_path] = actual_hash
            elif expected_sha256 is not None and self._search_index_hashes[index_path] != expected_sha256:
                raise ValueError("Index changed while the app was running. Restart the app after rebuilding.")
            return self._search_indexes[index_path]

    _RATING_CATEGORY_SQL = "CASE WHEN rating IN ('general','sensitive','questionable','explicit') THEN rating ELSE 'unclassified' END"

    @staticmethod
    def _normalize_catalog_ratings(ratings):
        allowed = {"general", "sensitive", "questionable", "explicit", "unclassified"}
        if ratings is None:
            return None
        if not isinstance(ratings, list) or any(not isinstance(value, str) or value not in allowed for value in ratings):
            raise SearchInputError("ratings must contain supported image categories")
        normalized = tuple(sorted(set(ratings)))
        return None if set(normalized) == allowed else normalized

    @classmethod
    def _catalog_rating_where(cls, con, normalized_ratings):
        if normalized_ratings is None:
            return "", []
        if not normalized_ratings:
            return " WHERE 0", []
        columns = {row[1] for row in con.execute("PRAGMA table_info(image_meta)")}
        if "rating" not in columns:
            return ("", []) if "unclassified" in normalized_ratings else (" WHERE 0", [])
        placeholders = ','.join('?' for _ in normalized_ratings)
        return f" WHERE {cls._RATING_CATEGORY_SQL} IN ({placeholders})", list(normalized_ratings)

    def load_catalog_count(self, model_id: ModelId, ratings: list[str] | None = None) -> int:
        normalized = self._normalize_catalog_ratings(ratings)
        model_id = normalize_model_id(model_id)
        key = (model_id, normalized)
        if key in self._catalog_counts:
            return self._catalog_counts[key]
        with self._catalog_count_lock:
            if key in self._catalog_counts:
                return self._catalog_counts[key]
            directory = self._search_model_meta_dir(model_id)
            con = open_readonly_database(directory / "sqlite_image_meta.db")
            try:
                if normalized == ():
                    count = 0
                elif normalized is None:
                    count = int(con.execute("SELECT COUNT(*) FROM image_meta").fetchone()[0])
                else:
                    # The source database may have no rating index. Aggregate all
                    # five categories once in SQL; never transfer the catalog to
                    # Python or repeat a full scan for each filter combination.
                    if model_id not in self._catalog_rating_counts:
                        columns = {row[1] for row in con.execute("PRAGMA table_info(image_meta)")}
                        if "rating" in columns:
                            rows = con.execute(f"SELECT {self._RATING_CATEGORY_SQL}, COUNT(*) FROM image_meta GROUP BY 1")
                            histogram = {category: int(total) for category, total in rows}
                        else:
                            histogram = {"unclassified": self.load_catalog_count(model_id)}
                        self._catalog_rating_counts[model_id] = histogram
                        self._catalog_counts[(model_id, None)] = sum(histogram.values())
                    count = sum(self._catalog_rating_counts[model_id].get(category, 0) for category in normalized)
                self._catalog_counts[key] = count
                return count
            finally:
                con.close()

    def load_startup_image_page(
        self, model_id: ModelId, page: int, size: int, ratings: list[str] | None = None,
    ) -> tuple[list[ImageItem], dict[ImageId, pathlib.Path]] | None:
        """Load one browser page without retaining all database rows or loading FAISS."""
        if type(page) is not int or page < 0 or type(size) is not int or size < 1:
            raise ValueError("page must be non-negative and size must be positive integers")
        normalized_ratings = self._normalize_catalog_ratings(ratings)
        offset = page * size
        # SQLite offsets are signed 64-bit; a later page cannot contain any row.
        if offset > 2**63 - 1:
            return [], {}
        try:
            index_dir = self._search_model_meta_dir(model_id)
        except SearchInputError:
            return None
        con = open_readonly_database(index_dir / "sqlite_image_meta.db")
        try:
            columns = {row[1] for row in con.execute("PRAGMA table_info(image_meta)")}
            if not {"image_id", "image_path"} <= columns:
                raise ValueError("Image metadata must contain image_id and image_path columns.")
            tags_column = "image_tags" if "image_tags" in columns else "NULL AS image_tags"
            rating_column = "rating" if "rating" in columns else "NULL AS rating"
            where, parameters = self._catalog_rating_where(con, normalized_ratings)
            limit = min(size, 2**63 - 1)
            direction = "ASC"
            sql_offset = offset
            total = self._catalog_counts.get((normalize_model_id(model_id), normalized_ratings))
            if total is not None:
                if offset >= total:
                    return [], {}
                limit = min(limit, total - offset)
                reverse_offset = total - offset - limit
                if reverse_offset < offset and (normalized_ratings is not None or reverse_offset <= 1024):
                    # Tail jumps should read from the end of the path index.
                    direction, sql_offset = "DESC", reverse_offset
            anchor_key = (normalize_model_id(model_id), normalized_ratings)
            with self._catalog_anchor_lock:
                anchors = list(self._catalog_page_anchors.get(anchor_key, {}).items())
            chosen_anchor = None
            for rank, (anchor_path, anchor_id) in anchors:
                forward_distance = offset - rank
                if rank <= offset and forward_distance < sql_offset and (normalized_ratings is not None or forward_distance <= 1024):
                    chosen_anchor = ("ASC", anchor_path, anchor_id)
                    direction, sql_offset = "ASC", forward_distance
                reverse_distance = rank - (offset + limit - 1)
                if reverse_distance >= 0 and reverse_distance < sql_offset and (normalized_ratings is not None or reverse_distance <= 1024):
                    normalized_model = anchor_key[0]
                    if normalized_model not in self._catalog_null_path_counts:
                        self._catalog_null_path_counts[normalized_model] = int(con.execute("SELECT COUNT(*) FROM image_meta WHERE image_path IS NULL").fetchone()[0])
                    # SQLite row-value ranges omit NULL paths. Use the ordinary
                    # reverse scan when NULL rows could be part of this range.
                    if not self._catalog_null_path_counts[normalized_model]:
                        chosen_anchor = ("DESC", anchor_path, anchor_id)
                        direction, sql_offset = "DESC", reverse_distance
            if chosen_anchor is not None:
                _, anchor_path, anchor_id = chosen_anchor
                comparison = ">=" if direction == "ASC" else "<="
                where += (" AND " if where else " WHERE ") + f"(image_path, image_id) {comparison} (?, ?)"
                parameters.extend((anchor_path, anchor_id))
            elif normalized_ratings is None and direction == "ASC" and sql_offset > 1024:
                # Seek using only the covering image_path index before reading
                # metadata. OFFSET on the full projection otherwise looks up
                # millions of discarded table rows. Preserve image_id tie order
                # by expanding the boundary path group and offsetting inside it.
                boundary = con.execute("SELECT image_path FROM image_meta ORDER BY image_path LIMIT 1 OFFSET ?", (sql_offset,)).fetchone()
                if boundary is None:
                    return [], {}
                if boundary[0] is not None:
                    before = con.execute("""SELECT
                        (SELECT COUNT(*) FROM image_meta WHERE image_path < ?) +
                        (SELECT COUNT(*) FROM image_meta WHERE image_path IS NULL)
                    """, (boundary[0],)).fetchone()[0]
                    where, parameters = " WHERE image_path >= ?", [boundary[0]]
                    sql_offset -= int(before)
            parameters.extend((limit, sql_offset))
            rows = con.execute(f"""
                SELECT image_id, image_path, {tags_column}, {rating_column}
                FROM image_meta{where} ORDER BY image_path {direction}, image_id {direction} LIMIT ? OFFSET ?
            """, parameters).fetchall()
            if direction == "DESC":
                rows.reverse()
            if rows:
                with self._catalog_anchor_lock:
                    saved = self._catalog_page_anchors.setdefault(anchor_key, OrderedDict())
                    for rank, row in ((offset, rows[0]), (offset + len(rows) - 1, rows[-1])):
                        if row[0] is not None and row[1] is not None:
                            saved[rank] = (row[1], row[0])
                            saved.move_to_end(rank)
                    # Keep only boundaries of recently read pages, never the
                    # full catalog; nearby scrolling then uses indexed seeks.
                    while len(saved) > 256:
                        saved.popitem(last=False)
            items = []
            paths = {}
            for identifier, image_path, tags, rating in rows:
                image_id = ImageId(str(identifier))
                relative_path = pathlib.Path(str(image_path))
                items.append(ImageItem(
                    image_id, ImageName(str(relative_path)), ImageTags(tags or ""), rating=rating or "",
                ))
                paths[image_id] = relative_path
            return items, paths
        finally:
            con.close()

    @cache
    def load_startup_image_items(
        self,
        model_id: ModelId,
    ) -> tuple[list[ImageItem], dict[ImageId, pathlib.Path]]:
        """起動時に必要な画像一覧とImageId->相対パスをSQLiteから構築して返す。"""

        startup_items_by_id: dict[ImageId, ImageItem] = {}
        id_to_path: dict[ImageId, pathlib.Path] = {}
        try:
            db_path = self._search_model_meta_dir(model_id) / "sqlite_image_meta.db"
        except ValueError:
            self._logger.warning("起動モデルがローカル登録されていません: %s", model_id)
            return [], {}

        if not db_path.exists():
            self._logger.warning(
                "起動用のsqlite_image_meta.dbが見つかりません: %s",
                db_path,
            )
            return [], {}

        con = open_readonly_database(db_path)
        try:
            result: pd.DataFrame = pd.read_sql_query(
                """
                SELECT image_id, image_path, image_tags
                FROM image_meta
                """,
                con,
            )
        finally:
            con.close()

        for row in result.itertuples(index=False):
            image_id = ImageId(str(row.image_id))
            if image_id in startup_items_by_id:
                continue

            image_path = pathlib.Path(str(row.image_path))
            tags = row.image_tags if row.image_tags is not None else ""
            startup_items_by_id[image_id] = ImageItem(
                id=image_id,
                display_name=ImageName(str(image_path)),
                tags=ImageTags(tags),
            )
            id_to_path[image_id] = image_path

        startup_items = sorted(
            startup_items_by_id.values(),
            key=lambda item: item.display_name.name,
        )
        return startup_items, id_to_path

    @cache
    def get_mean_meta_vector(self, model_id: ModelId) -> np.ndarray | None:
        """ModelIdからimage meta全体の平均ベクトルを取得する。"""

        index_dir = self._search_model_meta_dir(model_id)
        index_path = index_dir / "metafiles.index"
        mean_path = pathlib.Path(str(index_path) + ".mean.npy")
        manifest = read_index_manifest(index_path)
        self._check_required_manifest(index_path, manifest)
        if not mean_path.is_file():
            if manifest is not None or pathlib.Path(str(index_path) + ".meta.json").is_file():
                raise ValueError("Mean vector is missing from a centered index. Rebuild the index.")
            self._logger.warning(
                "旧形式の非中心化FAISS indexとして検索します（mean/metadata sidecarなし）: %s", index_path
            )
            return None
        if manifest is not None:
            self._load_search_index(index_path, manifest["index_sha256"])
        with mean_path.open("rb") as stream:
            if manifest is not None:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != manifest["mean_sha256"]:
                    raise ValueError("Mean vector and index manifest belong to different builds. Rebuild the index.")
                stream.seek(0)
            mean_vector = np.load(stream, allow_pickle=False)
        if mean_vector.ndim != 1 or not np.isfinite(mean_vector).all():
            raise ValueError("Mean vector must be a finite one-dimensional array. Rebuild the index.")
        if manifest is not None and mean_vector.size != manifest["dimension"]:
            raise ValueError("Mean vector dimension does not match the index manifest. Rebuild the index.")
        return mean_vector
