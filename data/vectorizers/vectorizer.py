
from typing import Any
from abc import ABC, abstractmethod
import numpy as np

from ml.data.readers import DatasetReader

class Vectorizer(ABC):
    @abstractmethod
    def index_vectorizer(self, reader: DatasetReader):
        pass

    @abstractmethod
    def vectorize_input(self, input: Any) -> np.ndarray:
        pass

    @abstractmethod
    def vectorize_label(self, label: Any) -> np.ndarray:
        pass

    @abstractmethod
    def output_size(self) -> int:
        pass

    @abstractmethod
    def input_size(self) -> int:
        pass
