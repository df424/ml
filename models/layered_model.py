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
        for i, l in enumerate(self._layers):
            #print(f'LAYER{i} INPUT SHAPE: {h.shape}')
            h = l.predict(h)
            #print(f'LAYER{i} OUTPUT SHAPE: {h.shape}')
        return h

    def backward(self, loss: np.ndarray) -> None:
        # Starting with loss calcualted from the objective function propegate back.
        dl = loss
        for i, l in enumerate(reversed(self._layers)):
            #print(f'BACKPROP LAYER{len(self._layers)-i-1} INPUT SHAPE: {dl.shape}')
            dl = l.backward(dl)
            #print(f'BACKPROP LAYER{len(self._layers)-i-1} OUTPUT SHAPE: {dl.shape}')

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
