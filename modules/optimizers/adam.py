
from ml.modules.optimizers.optimizer import Optimizer
from ml.models import Model
import numpy as np

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._momentums = {}
        self._rms = {}
        self._iteration = 0

    def update(self, model: Model):
        self._iteration += 1

        # For each parameter in our model.
        for k, param in model.parameters.items():
            # if we have a gradient for the parameter.
            if k in model.gradients:
                # If this is the first iteration we might not have a momentum so create it as zero here.
                if k not in self._momentums:
                    self._momentums[k] = np.zeros(param.shape)
                    self._rms[k] = np.zeros(param.shape)

                # Get the gradient for the parameter.
                dp = model.gradients[k]

                # Calcualte the delta from the momentum.
                m_dp = self._beta1 * self._momentums[k] + (1-self._beta1) * dp
                r_dp = self._beta2 * self._rms[k] + (1-self._beta2) * dp**2

                # Do bias correction.
                m_dp_norm = m_dp / (1-self._beta1**self._iteration)
                r_dp_norm = r_dp / (1-self._beta2**self._iteration)

                # Store the momentum for next update.
                self._momentums[k] = m_dp
                self._rms[k] = r_dp
                
                # Actually update our parameters.
                param -= self._lr * m_dp_norm / (np.sqrt(r_dp_norm) + self._epsilon)
