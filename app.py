import pathlib
# from app.infrastructure.dummy_repository import DummyRepository
# from app.infrastructure.dummy_accessor import DummyAccessor

from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.local_repository import LocalRepository

from app.presentation.controller import start_app
from app.application import usecase

def main():

    #コンフィグ
    image_dir_path: pathlib.Path =pathlib.Path('C:/Users/ikuto/projects/test_dataset/pokemon2')
    meta_dir_path: pathlib.Path=pathlib.Path('C:/Users/ikuto/projects/test_dataset/meta')

    #TODO 使用するパッケージをインストールすることが手間であるため、ダミーを使用している。
    accessor: LocalAccessor = LocalAccessor(meta_dir_path)
    # accessor = DummyAccessor()

    repository: LocalRepository = LocalRepository(image_dir_path)
    # repository = DummyRepository()

    in_usecase: usecase.Usecase = usecase.Usecase(repository, accessor)
    start_app(in_usecase)

if __name__ == "__main__":
    main()
