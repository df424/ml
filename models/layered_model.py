from typing import Dict, Any
from abc import ABC, abstractmethod, abstractproperty
import numpy as np

from ml.models import Model

class LayeredModel(Model):
    def __init__(self):
        self._layers = []

    def add_layer(self, layer: Model):
        self._layers.append(layer)

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = X
        for l in self._layers:
            h = l.predict(h)
        return h

    def backward(self, loss: np.ndarray) -> None:
        # Starting with loss calcualted from the objective function propegate back.
        dl = loss
        for l in reversed(self._layers):
            dl = l.backward(dl)

    @property
    def parameters(self) -> Dict[str, Any]:
        # TODO: Aggregate and return the parameters of all our layers.
        rv = {}
        for i, l in enumerate(self._layers):
            for k, p in l.parameters.items():
                rv[f'layer_{i}_{k}'] = p
        return rv

    @property
    def gradients(self) -> Dict[str, Any]:
        # TODO: Aggregate and return the gradients of all our layers.
        rv = {}
        for i, l in enumerate(self._layers):
            for k, p in l.gradients.items():
                rv[f'layer_{i}_{k}'] = p
        return rv
