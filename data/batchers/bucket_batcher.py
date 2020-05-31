
from typing import List, Generator, Tuple
import numpy as np
import itertools
import random

from ml.data.batchers import Batcher
from ml.data.readers import DatasetReader
from ml.data.vectorizers import Vectorizer
from ml.data import Instance

class BucketBatcher(Batcher):
    def __init__(self, 
        reader: DatasetReader, 
        vectorizer: Vectorizer,
        batch_size: int,
        shuffle: bool = False,
        cache_after_first: bool = False
        ):

        self._reader = reader
        self._vectorizer = vectorizer
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._cache_after_first = cache_after_first
        self._cache = []
        self._is_first = True

    def batches(self) -> Generator[Tuple[np.ndarray, np.ndarray, List[Instance]], None, None]:
        # If we are not caching, or we are caching but this is the first iteration...
        if not self._cache or self._is_first:
            instances = []
            for inst in self._reader.read():
                x, y = self._get_vectors(inst)
                instances.append((x, y, inst))

                if len(instances) == self._batch_size:
                    if self._shuffle:
                        random.shuffle(instances)
                    
                    if self._cache_after_first:
                        self._cache.extend(instances)

                    yield self._to_matrix(instances=instances)

                    # Reset the list of instances...
                    instances.clear()

            # hnadle the last batch which may exist, and may be partial...
            if len(instances) > 0:
                if self._cache_after_first:
                    self._cache.extend(instances)

                yield self._to_matrix(instances=instances)
            
            # Use cache from now on if we built it.
            self._is_first = False

        else: # Must be that we are caching and we already built the cache...
            # shuffle it if we need to...
            if self._shuffle:
                random.shuffle(self._cache)
                
            batch_stride = len(self._cache) if self._batch_size < 1 else self._batch_size

            for i in range(0, len(self._cache), batch_stride):
                yield self._to_matrix(self._cache[i:i+batch_stride])

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _to_matrix(self, instances: List[Tuple[np.ndarray, np.ndarray, Instance]]) -> Tuple[np.ndarray, np.ndarray, List[Instance]]:
        X, Y, insts = zip(*instances)
        return np.concatenate(X), np.concatenate(Y), list(insts)
            
    def _get_vectors(self, inst: Instance) -> Tuple[np.ndarray, np.ndarray]:
        if self._vectorizer:
            return self._vectorizer.vectorize_input(inst.x), self._vectorizer.vectorize_label(inst.y)
        else:
            return inst.x, inst.y
        