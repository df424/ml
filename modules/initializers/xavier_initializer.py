
from ml.modules.initializers import Initializer
import numpy as np

class XavierInitializer(Initializer):
    def initialize(self, weights: np.ndarray) -> np.ndarray:
        return np.random.randn(*weights.shape) * np.sqrt(1/weights.shape[0])