from typing import Literal, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from project.pipeline.balance_data import balance_data_indices_reduction, balance_data_indices_duplication


class DataLoader:
    def __init__(
            self,
            data_path: str = "../data/semeval/train_all_tasks.csv",
            batch_size: int = 4,
            shuffle: bool = False,
            seed: int = 42,
            balance_data_method: Literal["reduction", "duplication", "none"] = "none",
            task_column_name: Optional[str] = "label_sexist",
    ):
        # save vars
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.balance_data_method = balance_data_method
        self.task_column_name = task_column_name

        # load data
        self.df = pd.read_csv(data_path)

        # set seed
        np.random.seed(seed)

        # prepare indices
        self.indices = self._prepare_indices()

    def _prepare_indices(self) -> np.ndarray:
        if self.task_column_name is None:
            indices = np.arange(len(self.df))
            return indices

        # get the labels
        labels = self.df[self.task_column_name].values

        # balance the data
        if self.balance_data_method == "reduction":
            indices = balance_data_indices_reduction(labels, self.seed)
        elif self.balance_data_method == "duplication":
            indices = balance_data_indices_duplication(labels, self.seed)
        elif self.balance_data_method == "none":
            indices = np.arange(len(labels))
        else:
            raise ValueError(f"Unknown balance_data_method: {self.balance_data_method}")

        # return the indices
        return indices

    def __len__(self):
        len_iters = len(self.indices) // self.batch_size
        if len(self.indices) % self.batch_size != 0:
            len_iters += 1

        return len_iters

    def iter_batch_indices(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i + self.batch_size]

    def __iter__(self):
        for batch_indices in self.iter_batch_indices():
            yield self.df.iloc[batch_indices]


if __name__ == "__main__":
    data_loader = DataLoader(batch_size=4, balance_data_method="duplication", shuffle=True)
    for batch in tqdm(data_loader):
        print(batch)
        break
