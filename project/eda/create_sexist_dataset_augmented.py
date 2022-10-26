import pandas as pd

from project.pipeline.augmenter import TextAugmenter


def create_sexist_dataset_augmented():
    # load the original dataset
    df_original = pd.read_csv("../data/custom/train_sexism.csv")

    # load augmenter
    text_augmenter = TextAugmenter(device="cuda")

    # augment the dataset
    all_dfs = []
    for method in text_augmenter.get_augmentation_methods("medium"):
        print(f"Augmenting dataset with method: {method}")
        df_augmented = df_original.copy()
        df_augmented["text"] = text_augmenter(
            df_augmented["text"].values, method=method, progress_bar=True
        )
        all_dfs.append(df_augmented)

    # concatenate all the dfs
    df = pd.concat(all_dfs, ignore_index=True)

    # save the dataframe as csv
    df.to_csv("../data/custom/train_sexism_augmented.csv", index=False)


if __name__ == "__main__":
    create_sexist_dataset_augmented()