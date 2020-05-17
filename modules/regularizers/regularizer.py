
from abc import ABC, abstractmethod
import numpy as np

class Regularizer(ABC):
    @abstractmethod
    def regularize(self, weights: np.ndarray) -> None:
        pass