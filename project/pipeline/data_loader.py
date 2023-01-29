from typing import Literal, Optional, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from project.pipeline.balance_data import balance_data_indices_reduction, balance_data_indices_duplication


class DataLoader:
    def __init__(
            self,
            data_path: Union[List[str], str] = "../data/semeval/train_all_tasks.csv",
            batch_size: int = 4,
            shuffle: bool = False,
            seed: int = 42,
            balance_data_method: Literal["reduction", "duplication", "none"] = "none",
            task_column_name: Optional[str] = "label_sexist",
            exclude_values: List[str] = None,
    ):
        # save vars
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.balance_data_method = balance_data_method
        self.task_column_name = task_column_name
        self.exclude_values = exclude_values

        # if exclude_values is not None, set it to []
        if self.exclude_values is None:
            self.exclude_values = []

        # load data
        self.df = self._load_data()

        # Remove all the values from the exclude_values list
        if self.exclude_values:
            self.df = self.df[~self.df[self.task_column_name].isin(self.exclude_values)]
            self.df.reset_index(drop=True, inplace=True)

        # set seed
        np.random.seed(seed)

        # prepare indices
        self.indices = self._prepare_indices()

    def _load_data(self):
        # check if data_path is a str
        if isinstance(self.data_path, str):
            self.data_path = [self.data_path]

        # load the data
        dfs = []
        for data_path in self.data_path:
            df = pd.read_csv(data_path)
            dfs.append(df)

        # concat the data
        df = pd.concat(dfs, axis=0)

        # drop all instances without task_column_name
        if self.task_column_name is not None:
            df = df.dropna(subset=[self.task_column_name])

        # reset the indices
        df.reset_index(drop=True, inplace=True)

        return df

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
