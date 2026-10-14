import logging
import pathlib

from app.domain.domain_object import ModelId, ModelItem
from app.domain.errors import SearchInputError
from app.infrastructure.index_files import IndexFiles, model_item_from_directory

logger = logging.getLogger(__name__)


class ModelRegistry:
    """メタデータルート直下にある検索インデックスと、その検索モデルの対応。

    リクエストごとにディレクトリを走査しないよう、構築時に一度だけ走査する。
    インデックスを追加・更新した場合はアプリを再起動する。
    """

    def __init__(self, meta_dir: pathlib.Path) -> None:
        self._entries: dict[ModelId, tuple[ModelItem, IndexFiles]] = {}
        root = pathlib.Path(meta_dir).resolve()
        if not root.is_dir():
            return
        for directory in sorted(root.iterdir(), key=lambda path: path.name.lower()):
            files = IndexFiles(directory.resolve())
            if not files.has_search_index():
                continue
            # リンクなどでルート外を指すインデックスは登録しない。
            if any(not path.resolve().is_relative_to(root) for path in (files.directory, *files.all())):
                logger.warning("メタデータルート外を指すインデックスを無視します: %s", directory)
                continue
            item = model_item_from_directory(files.directory)
            self._entries.setdefault(item.id, (item, files))

    def items(self) -> list[ModelItem]:
        """登録されている検索モデルを表示名順に返す。"""
        return sorted((item for item, _ in self._entries.values()), key=lambda item: item.display_name.lower())

    def files(self, model_id: ModelId) -> IndexFiles:
        """ローカルに登録されたモデルのインデックスだけを解決する。"""
        entry = self._entries.get(model_id)
        if entry is None:
            # APIの応答に含まれるため、サーバー上の場所は書かない。
            raise SearchInputError(
                f"Search model is not registered: {model_id.model_name} ({model_id.pretrained or 'no pretrained tag'})"
            )
        return entry[1]
