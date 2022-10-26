import numpy as np
import pandas as pd


def balance_data_indices_reduction(
        labels: np.ndarray,
        seed: int = 42
) -> np.ndarray:
    """
    Balance the data by randomly sampling from the minority class.

    Args:
        labels (np.ndarray): The labels to balance.
        seed (int, optional): The seed to use for random sampling. Defaults to 42.

    Returns:
        np.ndarray: The indices to use for balancing the data.
    """

    # set seed
    np.random.seed(seed)

    # get the unique labels
    unique_labels = np.unique(labels)

    # get the counts of each label
    counts = np.array([np.sum(labels == label) for label in unique_labels])

    # get the minimum count
    min_count = np.min(counts)

    # create a list to store the balanced data
    balanced_data_indices = []

    # iterate the unique labels
    for label in unique_labels:
        # get the indices of the label
        indices = np.where(labels == label)[0]

        # randomly sample the indices
        sampled_indices = np.random.choice(indices, min_count, replace=False)

        # append the sampled indices to the balanced data
        balanced_data_indices.extend(sampled_indices)

    # return the balanced data indices
    return np.array(balanced_data_indices)


def balance_data_indices_duplication(
        labels: np.ndarray,
        seed: int = 42
) -> np.ndarray:
    """
    Balance the data by duplicating the minority classes.

    Args:
        labels (np.ndarray): The labels to balance.
        seed (int, optional): The seed to use for random sampling. Defaults to 42.

    Returns:
        np.ndarray: The indices to use for balancing the data.
    """

    # set seed
    np.random.seed(seed)

    # get the unique labels
    unique_labels = np.unique(labels)

    # get the counts of each label
    counts = np.array([np.sum(labels == label) for label in unique_labels])

    # get the minimum count
    max_count = np.max(counts)

    # create a list to store the balanced data
    balanced_data_indices = []

    # iterate the unique labels
    for label in unique_labels:
        # get the indices of the label
        indices = np.where(labels == label)[0]

        # get the number of times to duplicate the data
        num_duplications = max_count // len(indices) + 1

        # duplicate the data
        duplicated_indices = np.tile(indices, num_duplications)

        # randomly sample the indices
        sampled_indices = np.random.choice(duplicated_indices, max_count, replace=False)

        # append the sampled indices to the balanced data
        balanced_data_indices.extend(sampled_indices)

    # return the balanced data indices
    return np.array(balanced_data_indices)


if __name__ == "__main__":
    # load the data
    data_path = "../data/custom/train_sexism.csv"
    df = pd.read_csv(data_path)

    # print the counts
    print(df['label_sexist'].value_counts())

    # get the labels
    labels = df['label_sexist'].values

    # balance the data
    balanced_df_indices = balance_data_indices_duplication(labels)
    balanced_df = df.iloc[balanced_df_indices]

    # print the shape
    print(balanced_df.shape)

    # print the counts
    print(balanced_df['label_sexist'].value_counts())