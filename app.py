import pathlib

from app.presentation.controller import startApp
from app.application import usecase
#from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.dummy_accessor import DummyAccessor
from app.infrastructure.local_repository import LocalRepository

def main():

    #コンフィグ
    image_dir_path: pathlib.Path =pathlib.Path('./images')
    meta_dir_path=pathlib.Path('./meta')

    #TODO Accessorに使用するパッケージをインストールすることが手間であるため、ダミーを使用している。
    #accessor = LocalAccessor(meta_dir_path)
    accessor = DummyAccessor()

    repository = LocalRepository(image_dir_path)
    in_usecase = usecase.Usecase(repository, accessor)
    startApp(in_usecase)

if __name__ == "__main__":
    main()
