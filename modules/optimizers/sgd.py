
from ml.modules.optimizers.optimizer import Optimizer
from ml.models import Model
import numpy as np

class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0):
        self._lr = learning_rate
        self._beta = beta
        self._momentums = {}
        self._iteration = 1

    def update(self, model: Model):
        # For each parameter in our model.
        for k, param in model.parameters.items():
            # if we have a gradient for the parameter.
            if k in model.gradients:
                # If this is the first iteration we might not have a momentum so create it as zero here.
                if k not in self._momentums:
                    self._momentums[k] = np.zeros(param.shape)

                # Get the gradient for the parameter.
                dp = model.gradients[k]

                # Calcualte the delta from the momentum.
                V_dp = self._beta * self._momentums[k] + (1-self._beta) * dp

                # Do bias correction.
                V_dp = V_dp / (1-self._beta**self._iteration)
                self._iteration += 1

                # Store the momentum for next update.
                self._momentums[k] = V_dp

                # Actually update our parameters.
                param -= self._lr * V_dp
