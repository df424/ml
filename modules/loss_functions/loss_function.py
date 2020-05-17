
from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    @abstractmethod
    def scaler_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        pass

