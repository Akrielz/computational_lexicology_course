import pandas as pd

from project.pipeline.augmenter import TextAugmenter


def create_sexist_dataset_augmented():
    # load the original dataset
    df_original = pd.read_csv("../data/train_sexism.csv")

    # load augmenter
    text_augmenter = TextAugmenter()

    # augment the dataset
    all_dfs = []
    for method in text_augmenter.augmentation_methods:
        df_augmented = df_original.copy()
        df_augmented["comment_text"] = df_augmented["comment_text"].apply(
            lambda x: text_augmenter(x, method=method)
        )
        all_dfs.append(df_augmented)

    # concatenate all the dfs
    df = pd.concat(all_dfs, ignore_index=True)

    # save the dataframe as csv
    df.to_csv("../data/train_sexism_augmented.csv", index=False)


if __name__ == "__main__":
    create_sexist_dataset_augmented()