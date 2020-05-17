
from typing import Callable
from glob import glob
import math
import random


def split_by_filename(
    glob_expression: str, 
    label_extractor: Callable[[str], str],
    train_dev_split: float = 2.0/3.0
    ):

    # Group the files based on label so we can split them according to their priors.
    file_paths = glob(glob_expression)

    by_labels = {}
    for file_path in file_paths:
        label = label_extractor(file_path)

        if label not in by_labels: 
            by_labels[label] = []
        by_labels[label].append(file_path)

    # now we can randomly sample from each label group according to our split requirements...
    train_files = []
    dev_files = []
    for file_group in by_labels.values():
        n_test = math.ceil(len(file_group) * train_dev_split)
        random.shuffle(file_group)
        train_files.extend(file_group[:n_test])
        dev_files.extend(file_group[n_test:])

    # randomize the test set to start for fun.
    random.shuffle(train_files)

    return train_files, dev_files
