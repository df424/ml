
from typing import Callable, List, Dict
from tqdm.autonotebook import tqdm
import numpy as np
import sys

from ml.models import Model
from ml.data.batchers import Batcher
from ml.modules.loss_functions import LossFunction
from ml.modules.optimizers import Optimizer

class Trainer:
    def __init__(self, 
        model: Model, 
        loss: LossFunction,
        train_batcher: Batcher,
        optimizer: Optimizer,
        dev_batcher: Batcher=None,
        epochs: int = 1,
        metrics: List[Callable[[np.ndarray, np.ndarray], float]] = [],
        patience: int = sys.maxsize
    ):
        self._model = model
        self._loss = loss
        self._train_batcher = train_batcher
        self._dev_batcher = dev_batcher
        self._epochs = epochs
        self._optimizer = optimizer
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

            # Only run the dev set if we were given a dev set batcher to work with.
            if self._dev_batcher:
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

            # Create the description string for the progress bar.
            description = f'TRAINING: Epoch=[{i+1}/{self._epochs}], Training Loss={self._train_losses[-1]:.3f}'
            # If we have a dev set print its loss as well.
            if self._dev_batcher:
                description += f', Dev Loss={self._dev_losses[-1]:.3f}'
            # Actually update the progress bar. 
            pbar.set_description(description)
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
            epoch_loss += self._loss.scaler_loss(y=Y, y_hat=Y_hat) * batch_size

            # Other metrics.
            for mf in self._metric_functions:
                epoch_metrics[mf.__name__] += mf(Y, Y_hat) * batch_size

            if(update_model):
                self._model.backward(self._loss.gradient(Y, Y_hat))
                self._optimizer.update(self._model)

        losses.append(epoch_loss/sample_count)

        for mf in self._metric_functions:
            metrics[mf.__name__].append(epoch_metrics[mf.__name__]/sample_count)
            