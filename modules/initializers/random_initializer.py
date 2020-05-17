
from ml.modules.initializers import Initializer
import numpy as np

class RandomInitializer(Initializer):
    def __init__(self, minval: float=-1, maxval: float=1):
        assert maxval > minval
        self._minval = minval
        self._maxval = maxval

    def initialize(self, weights: np.ndarray) -> np.ndarray:
        delta = self._maxval - self._minval
        return np.random.rand(*weights.shape) * delta + self._minval
