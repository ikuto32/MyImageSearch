import asyncio
from functools import cache
import hashlib
import io
import itertools
import os
import pathlib
import mimetypes
from typing import List
import concurrent.futures
import zipfile

import tqdm

from app.domain.domain_object import (
    ImageItem,
    ImageId,
    Image,
    ImageName,
    ImageTags,
    ModelId,
    ModelItem,
    ModelName,
)
from app.domain.repository import Repository

from PIL import Image as PILImage

import open_clip


class LocalRepository(Repository):
    """ローカル上のファイルを対象としたRepository"""

    def __init__(self, image_dir_path: pathlib.Path) -> None:

        self._image_dir_path: pathlib.Path = image_dir_path
        self._id_to_path: dict[ImageId, pathlib.Path] = {}

    @cache
    def load_all_image_item(self) -> List[ImageItem]:
        async def _load_all_image_item_async():
            files = []
            # extensions = list(PILImage.registered_extensions().keys())
            extensions = ['.avif', '.avifs', '.blp', '.bmp', '.dib', '.bufr', '.cur', '.pcx', '.dcx', '.dds', '.ps', '.eps', '.fit', '.fits', '.fli', '.flc', '.ftc', '.ftu', '.gbr', '.gif', '.grib', '.h5', '.hdf', '.png', '.apng', '.jp2', '.j2k', '.jpc', '.jpf', '.jpx', '.j2c', '.icns', '.ico', '.im', '.iim', '.jfif', '.jpe', '.jpg', '.jpeg', '.mpg', '.mpeg', '.tif', '.tiff', '.mpo', '.msp', '.palm', '.pcd', '.pxr', '.pbm', '.pgm', '.ppm', '.pnm', '.pfm', '.psd', '.qoi', '.bw', '.rgb', '.rgba', '.sgi', '.ras', '.tga', '.icb', '.vda', '.vst', '.webp', '.wmf', '.emf', '.xbm', '.xpm']
            ext_set = set(e.lower() if e.startswith('.') else f'.{e.lower()}' for e in extensions)
            loop = asyncio.get_event_loop()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=320)
            queue = asyncio.Queue()
            await queue.put(str(self._image_dir_path))

            progress = tqdm.tqdm(total=0, unit="file", dynamic_ncols=True)
            update_interval = 100  # 進捗バーの更新間隔
            counter = 0

            async def worker():
                nonlocal counter
                while True:
                    current_dir = await queue.get()
                    try:
                        # ディレクトリエントリを非同期に取得
                        def list_dir(path):
                            with os.scandir(path) as it:
                                return [entry for entry in it]
                        entries = await loop.run_in_executor(executor, list_dir, current_dir)

                        for entry in entries:
                            try:
                                if entry.is_dir(follow_symlinks=False):
                                    await queue.put(entry.path)
                                elif entry.is_file(follow_symlinks=False):
                                    # ファイルの拡張子をチェック
                                    name_lower = entry.name.lower()
                                    if any(name_lower.endswith(ext) for ext in ext_set):
                                        files.append(entry.path)
                                        counter += 1
                                        if counter >= update_interval:
                                            progress.update(counter)
                                            counter = 0
                            except OSError:
                                # エントリへのアクセスに失敗した場合
                                continue
                    except PermissionError:
                        pass
                    finally:
                        queue.task_done()

            # ワーカータスクを起動
            tasks = []
            num_workers = 640  # ワーカー数は環境に合わせて調整
            for _ in range(num_workers):
                task = asyncio.create_task(worker())
                tasks.append(task)

            # キューが空になるまで待機
            await queue.join()

            # 残ったカウンタを更新
            if counter > 0:
                progress.update(counter)

            # ワーカータスクをキャンセル
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            progress.close()
            executor.shutdown()
            return files

        # 非同期関数を実行してファイルリストを取得
        loop = asyncio.get_event_loop()
        files = loop.run_until_complete(_load_all_image_item_async())

        # ImageItemを作成する関数
        def create_image_item(file_path: str):
            relative_file = os.path.relpath(file_path, str(self._image_dir_path))
            id = ImageId(hashlib.sha256(relative_file.encode()).hexdigest())
            return id, relative_file, ImageItem(id, ImageName(relative_file), ImageTags(relative_file))

        # ファイルを並列に処理してImageItemを作成
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(create_image_item, tqdm.tqdm(files, desc="ImageItemを作成中")))

        # self._id_to_pathを更新
        id_to_path = {id: relative_file for id, relative_file, _ in results}
        self._id_to_path.update(id_to_path)

        # ImageItemを抽出してソート
        items = [item for _, _, item in results]
        items.sort(key=lambda item: item.display_name.name)

        return items

    @cache
    def load_all_model_item(self) -> list[ModelItem]:

        items: list[ModelItem] = [
            ModelItem(
                ModelId(model_name, dataset), ModelName(f"{model_name}-{dataset}")
            )
            for (model_name, dataset) in open_clip.list_pretrained()
        ]
        return items

    @cache
    def load_image(self, image_id: ImageId) -> Image:
        relative_path = self._id_to_path.get(image_id)

        if relative_path is None:
            print(f"指定されたImageIdが存在しません: {image_id}")
            return None

        path: pathlib.Path = self._image_dir_path / relative_path
        binary: bytes = path.read_bytes()

        content_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'

        return Image(binary, content_type)

    @cache
    def load_small_image(self, image_id: ImageId) -> Image:
        relative_path = self._id_to_path.get(image_id)

        if relative_path is None:
            print(f"指定されたImageIdが存在しません: {image_id}")
            return None

        path: pathlib.Path = self._image_dir_path / relative_path

        # open image with PIL and resize to long side 400 px
        with PILImage.open(path) as img:
            img.thumbnail((400, 400))
            buffer = io.BytesIO()
            format = img.format if img.format else 'PNG'
            img.save(buffer, format=format)
            binary = buffer.getvalue()

        content_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'

        return Image(binary, content_type)

    def create_zip_from_images(self, images_with_names: list[tuple[Image, ImageName]]):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for image, image_name in images_with_names:
                extension = image.content_type.split('/')[-1]
                filename = f"{image_name.name}"
                zip_file.writestr(filename, image.binary)
        zip_buffer.seek(0)
        return zip_buffer

    @cache
    def get_image_name(self, image_id: ImageId) -> ImageName:
        relative_path = self._id_to_path.get(image_id)
        if relative_path is None:
            raise ValueError(f"指定されたImageIdが存在しません: {image_id}")

        # パスからファイル名だけを取得
        image_name = pathlib.Path(relative_path).name

        return ImageName(name=image_name)
