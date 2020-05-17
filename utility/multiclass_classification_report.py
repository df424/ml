
from typing import List
import numpy as np
import pandas as pd

class MultiClassClassificationReport():
    def __init__(self, Y: np.ndarray, Y_hat: np.ndarray, label_map: List[str]):
        self._labels = label_map
        self._Y = Y
        self._Y_hat = Y_hat
        self._confusion_matrix = MultiClassClassificationReport._build_confusion_matrix(self._Y, self._Y_hat)
        self._recall = np.diag(self._confusion_matrix)/np.sum(self._confusion_matrix, axis=1)
        self._precision = np.diag(self._confusion_matrix)/np.sum(self._confusion_matrix, axis=0)
        self._avg_recall = self._recall.mean()
        self._avg_precision = self._precision.mean()
        self._f1 = 2 * (self._precision * self._recall)/(self._precision+self._recall)
        self._f1_macro = self._f1.mean()
        self._accuracy = np.diag(self._confusion_matrix).sum()/self._confusion_matrix.sum()
    
    @property
    def confusion_matrix(self) -> np.ndarray:
        return pd.DataFrame(
            self._confusion_matrix,
            index=self._labels,
            columns=self._labels,
        ).sort_index().T.sort_index().T

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def recall(self) -> np.ndarray:
        return self._recall

    @property
    def precision(self) -> np.ndarray:
        return self._precision

    @property
    def f1(self) -> np.ndarray:
        return self._f1

    @property
    def macro_recall(self) -> float:
        return self._avg_recall

    @property
    def macro_precision(self) -> float:
        return self._avg_precision

    @property
    def macro_f1(self) -> float:
        return self._f1_macro

    def per_class_stats(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.concatenate([
                self.recall.reshape(1,-1),
                self.precision.reshape(1,-1),
                self.f1.reshape(1,-1),
            ]).T,
            index=self._labels,
            columns=['recall', 'precision', 'f1']
        ).sort_index()

    @staticmethod
    def _build_confusion_matrix(Y: np.ndarray, Y_hat: np.ndarray):
        labels = Y.argmax(axis=1)
        preds = Y_hat.argmax(axis=1)

        matrix = np.zeros((labels.max()+1, labels.max()+1))

        for l, p in zip(labels, preds):
            matrix[p, l] += 1

        return matrix
