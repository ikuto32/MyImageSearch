import pathlib

from app.presentation.controller import start_app
from app.application import usecase
from app.infrastructure.local_repository import LocalRepository

def main():
    repository = LocalRepository(image_dir_path=pathlib.Path('./images'),meta_dir_path=pathlib.Path('./meta'))
    in_usecase = usecase.Usecase(repository)
    start_app(in_usecase)

if __name__ == "__main__":
    main()
