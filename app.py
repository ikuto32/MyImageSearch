import pathlib

# from app.infrastructure.dummy_repository import DummyRepository
# from app.infrastructure.dummy_accessor import DummyAccessor

from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.local_repository import LocalRepository
from app.domain.domain_object import ModelId

from app.presentation.controller import start_app
from app.application import usecase


def main():

    # コンフィグ
    image_dir_path: pathlib.Path = pathlib.Path('//192.168.1.46/ikutoDataset/dataset/gallery-dl')
    meta_dir_path: pathlib.Path = pathlib.Path('C:/Users/ikuto/projects/clip_meta')

    accessor: LocalAccessor = LocalAccessor(meta_dir_path)

    repository: LocalRepository = LocalRepository(image_dir_path)
    startup_model_id = ModelId("ViT-L-14", "openai")

    in_usecase: usecase.Usecase = usecase.Usecase(
        repository, accessor, startup_model_id
    )
    start_app(in_usecase)


if __name__ == "__main__":
    main()
