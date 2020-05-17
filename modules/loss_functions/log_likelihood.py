

import numpy as np
import sys

from ml.modules.activation_functions import sigmoid
from ml.modules.loss_functions import LossFunction

class SigmoidLogLikelihood(LossFunction):
    def scaler_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        # pylint: disable=assignment-from-no-return
        norm_y_hat = np.maximum(y_hat, sys.float_info.epsilon)
        norm_one_minus_y_hat = np.maximum(1-y_hat, sys.float_info.epsilon)
        loss = -(y * np.log(norm_y_hat) + (1-y) * np.log(norm_one_minus_y_hat)).sum(axis=1).mean()
        return loss

    def gradient(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return -(y-y_hat)