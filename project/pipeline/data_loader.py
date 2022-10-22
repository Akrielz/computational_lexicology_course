import pandas as pd


class DataLoader:
    def __init__(
            self,
            data_path: str = "../data/train_all_tasks.csv",
            batch_size: int = 4,
    ):
        self.data_path = data_path
        self.batch_size = batch_size

        self.df = pd.read_csv(data_path)

    def __len__(self):
        len_iters = len(self.df) // self.batch_size
        if len(self.df) % self.batch_size != 0:
            len_iters += 1
        return len_iters

    def __iter__(self):
        for i in range(len(self)):
            yield self.df.iloc[i * self.batch_size:(i + 1) * self.batch_size]


if __name__ == "__main__":
    data_loader = DataLoader()
    for batch in data_loader:
        print(batch['text'].values)
        break
