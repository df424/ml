
import numpy as np

def class_accuracy(Y: np.array, Y_hat: np.array) -> float:
    return (Y.argmax(axis=1) == Y_hat.argmax(axis=1)).mean()