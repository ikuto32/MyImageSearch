from app.application.accessor import Accessor
from app.application.embedding_backend import SearchEmbeddingBackend
from app.application.search_index import SearchIndex
from app.domain.domain_object import ModelId
from app.infrastructure.model_registry import ModelRegistry
from app.infrastructure.once_cache import OnceCache
from app.infrastructure.search_index_loader import load_search_index


def create_embedding_backend(model_id: ModelId) -> SearchEmbeddingBackend:
    # 推論ライブラリの読み込みは重いため、最初の検索まで遅らせる。
    from app.infrastructure import embedding_backends

    return embedding_backends.create_embedding_backend(model_id)


class LocalAccessor(Accessor):
    """ローカルに登録されたインデックスと埋め込みモデルを読み込み、プロセス内で共有するAccessor"""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._search_indexes: OnceCache[ModelId, SearchIndex] = OnceCache()
        self._embedding_backends: OnceCache[ModelId, SearchEmbeddingBackend] = OnceCache()

    def load_search_index(self, model_id: ModelId) -> SearchIndex:
        """同時検索でも同一モデルの巨大なFAISS indexとメタデータを一度だけ読み込む。"""
        # 未登録のモデルIDをキャッシュのキーとして保持しないよう、先に登録を確認する。
        files = self._registry.files(model_id)
        return self._search_indexes.get(model_id, lambda: load_search_index(files))

    def load_embedding_backend(self, model_id: ModelId) -> SearchEmbeddingBackend:
        """初回の並列リクエストでも同一モデルを一度だけロードする。"""
        self._registry.files(model_id)
        return self._embedding_backends.get(model_id, lambda: create_embedding_backend(model_id))
