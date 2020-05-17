
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, loss: np.ndarray, alpha: float) -> None:
        pass
