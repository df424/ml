
from abc import ABC, abstractmethod
import numpy as np

class Initializer(ABC):
    @abstractmethod
    def initialize(self, weights: np.ndarray) -> np.ndarray:
        pass
