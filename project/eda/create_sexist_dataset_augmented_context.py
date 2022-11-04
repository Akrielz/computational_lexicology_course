import pandas as pd
from tqdm import tqdm

from project.pipeline.augmentations import load_context_aug
from project.pipeline.data_loader import DataLoader


def create_sexist_dataset_augmented():
    # load the original dataset
    data_loader = DataLoader(data_path="../data/custom/train_sexist.csv", batch_size=16, shuffle=False)

    context_aug = load_context_aug(device="cuda")

    augmented_texts = []
    labels = []
    for batch in tqdm(data_loader):
        text = list(batch["text"].values)
        augmented_text = context_aug.augment(text)
        augmented_texts.append(augmented_text)

        label = list(batch["label_sexist"].values)
        labels.append(label)

    # flatten the list
    augmented_texts = [item for sublist in augmented_texts for item in sublist]
    labels = [item for sublist in labels for item in sublist]

    # create a dataframe
    df = pd.DataFrame({"text": augmented_texts, "label_sexist": labels})

    # save the dataframe as csv
    df.to_csv("../data/custom/train_sexist_augmented_0.csv", index=False)


if __name__ == "__main__":
    create_sexist_dataset_augmented()
