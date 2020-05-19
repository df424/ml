
from ml.modules.initializers import Initializer
import numpy as np

class HeInitializer(Initializer):
    def initialize(self, weights: np.ndarray) -> np.ndarray:
        return np.random.randn(*weights.shape) * np.sqrt(2/weights.shape[0])