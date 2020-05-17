

import numpy as np
import sys

from ml.modules.loss_functions.loss_function import LossFunction
from ml.modules.activation_functions import softmax

class CrossEntropyLoss(LossFunction):
    def scaler_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        # pylint: disable=assignment-from-no-return
        norm_y_hat = np.maximum(y_hat, sys.float_info.epsilon)
        loss = -(y*np.log(norm_y_hat)).sum()
        return loss

    def gradient(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return y_hat-y