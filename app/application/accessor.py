from abc import ABC, abstractmethod

from app.application.embedding_backend import SearchEmbeddingBackend
from app.application.search_index import SearchIndex
from app.domain.domain_object import ModelId


class Accessor(ABC):
    """キャッシュなどドメイン(アプリケーションの役割)と直接関係ない外部接続を行うインターフェイス"""

    @abstractmethod
    def load_search_index(self, model_id: ModelId) -> SearchIndex:
        """ModelIdから検索インデックスと、その行番号に対応するメタデータを取得する。"""

    @abstractmethod
    def load_embedding_backend(self, model_id: ModelId) -> SearchEmbeddingBackend:
        """ModelIdから検索クエリ生成用の埋め込みバックエンドを取得する。"""
