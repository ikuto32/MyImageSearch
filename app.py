import pathlib

from app.presentation.controller import start_app
from app.application import usecase
#from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.dummy_accessor import DummyAccessor
from app.infrastructure.local_repository import LocalRepository

def main():
<<<<<<< HEAD
    repository = LocalRepository(image_dir_path=pathlib.Path('./images'),meta_dir_path=pathlib.Path('./meta'))
    in_usecase = usecase.Usecase(repository)
    start_app(in_usecase)
=======

    #コンフィグ
    image_dir_path: pathlib.Path =pathlib.Path('./images')
    meta_dir_path=pathlib.Path('./meta')

    #TODO Accessorに使用するパッケージをインストールすることが手間であるため、ダミーを使用している。
    #accessor = LocalAccessor(meta_dir_path)
    accessor = DummyAccessor()

    repository = LocalRepository(image_dir_path)
    in_usecase = usecase.Usecase(repository, accessor)
    startApp(in_usecase)
>>>>>>> 51a1dac8761227a79e38a52f59cf35b7bc64f9ee

if __name__ == "__main__":
    main()
