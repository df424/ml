
from typing import Dict, Any
import numpy as np
from ml.models import Model

class FlattenLayer(Model):
    def __init__(self):
        self._cache = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        # We need to keep X so we can resture the shape on the backwards pass.
        self._cache['input'] = X
        self._cache['old_shape'] = X.shape        
        rv = X.reshape(X.shape[0], -1)
        self._cache['output'] = rv
        return rv

    def backward(self, loss: np.ndarray) -> None:
        return loss.reshape(self._cache['old_shape'])

    @property
    def parameters(self) -> Dict[str, Any]:
        return {}

    @property
    def gradients(self) -> Dict[str, Any]:
        return {}
