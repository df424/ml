
from typing import Callable, List, Dict
from ml.models import Model
from ml.data.batchers import Batcher
from ml.modules.loss_functions import log_likelihood_sigmoid_update
from tqdm.autonotebook import tqdm
import numpy as np
import sys

class Trainer:
    def __init__(self, 
        model: Model, 
        loss: Callable[[np.ndarray, np.ndarray], float],
        update: Callable[[np.ndarray, np.ndarray, np.ndarray], None],
        train_batcher: Batcher,
        alpha: float,
        dev_batcher: Batcher=None,
        epochs: int = 1,
        metrics: List[Callable[[np.ndarray, np.ndarray], float]] = [],
        patience: int = sys.maxsize
    ):
        self._model = model
        self._loss = loss
        self._update = update
        self._train_batcher = train_batcher
        self._dev_batcher = dev_batcher
        self._epochs = epochs
        self._alpha = alpha
        self._train_losses = []
        self._dev_losses = []
        self._metric_functions = metrics
        self._train_metrics: Dict[str, List[float]] = {m.__name__:[] for m in metrics}
        self._dev_metrics: Dict[str, List[float]] = {m.__name__:[] for m in metrics}
        self._patience = patience

    @property
    def training_metrics(self) -> Dict[str, List[float]]:
        return self._train_metrics

    @property
    def dev_metrics(self) -> Dict[str, List[float]]:
        return self._dev_metrics

    @property
    def training_loss(self) -> List[float]:
        return self._train_losses

    @property
    def dev_loss(self) -> List[float]:
        return self._dev_losses

    def train(self):
        patience_counter = 0
        min_dev_loss = sys.float_info.max

        pbar = tqdm(range(self._epochs))
        for i in pbar:

            self._train(
                batcher=self._train_batcher,
                losses=self._train_losses,
                metrics=self._train_metrics,
                update_model=True,
            )

            self._train(
                batcher=self._dev_batcher,
                losses=self._dev_losses,
                metrics=self._dev_metrics,
                update_model=False,
            )

            # Handle patience stuff to abort training early...
            if min_dev_loss <= self._dev_losses[-1]:
                patience_counter += 1
            else:
                patience_counter = 0
                min_dev_loss = self._dev_losses[-1]

            if(patience_counter >= self._patience):
                print(f'Out of patience.  Dev loss has not decreased in {self._patience} epochs.')
                return

            pbar.set_description(f'TRAINING: Epoch=[{i+1}/{self._epochs}], Training Loss={self._train_losses[-1]:.3f}, Dev Loss={self._dev_losses[-1]:.3f}')
            pbar.update()
    
    def _train(self, 
        batcher: Batcher, 
        losses:List[float], 
        metrics: Dict[str, List[float]], 
        update_model: bool,
    ):
        epoch_loss = 0
        sample_count = 0

        # calculate metrics for this epoch.
        epoch_metrics = {mf.__name__:0 for mf in self._metric_functions}

        for batch in batcher.batches():
            # break out the batch data.
            X, Y, _ = batch
            batch_size = X.shape[0]

            # keep track of how many samples weve added..
            sample_count += batch_size

            # make a prediction.
            Y_hat = self._model.predict(X)

            # Update all our metrics.
            # calculate the loss...
            epoch_loss += self._loss(y=Y, y_hat=Y_hat) * batch_size

            # Other metrics.
            for mf in self._metric_functions:
                epoch_metrics[mf.__name__] += mf(Y, Y_hat) * batch_size

            if(update_model):
                self._model.update(
                    gradients=self._update(X, Y, Y_hat), 
                    alpha=self._alpha
                )

        losses.append(epoch_loss/sample_count)

        for mf in self._metric_functions:
            metrics[mf.__name__].append(epoch_metrics[mf.__name__]/sample_count)
            