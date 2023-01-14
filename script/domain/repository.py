

from abc import abstractmethod
from abc import ABCMeta
import numpy as np


class Repository(metaclass=ABCMeta):

    @abstractmethod
    def load_image_bytes(path : str) -> bytes:

        pass

    def load_meta(path : str) -> np.ndarray:

        pass

    