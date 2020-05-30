
from typing import Dict, Any
from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class Model(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, loss: np.ndarray) -> None:
        pass

    @abstractproperty
    def parameters(self) -> Dict[str, Any]:
        pass

    @abstractproperty
    def gradients(self) -> Dict[str, Any]:
        pass
