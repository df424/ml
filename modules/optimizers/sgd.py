
from ml.modules.optimizers.optimizer import Optimizer
from ml.models import Model

class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        self._lr = learning_rate

    def update(self, model: Model):
        for k, param in model.parameters.items():
            if k in model.gradients:
                dp = model.gradients[k]
                param -= self._lr * dp
