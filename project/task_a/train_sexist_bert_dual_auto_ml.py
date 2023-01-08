from typing import Optional

import numpy as np
import pandas as pd
import torch
from pycaret.classification import compare_models, save_model, setup, predict_model, load_model
from tqdm import tqdm

from project.pipeline.classic_tokenizers import get_bert_tokenizer
from project.pipeline.data_loader import DataLoader
from project.task_a.train_sexist_bert_pretrained_dual import build_model


def create_embeddings():
    # get tokenizer
    tokenizer = get_bert_tokenizer()

    # get model
    model = build_model(model_path="../trained_agents/sexist_bert_pretrained_dual_a_original.pt")
    model.eval()

    # remove the classifier to get the embeddings
    model.classifier = torch.nn.Identity()

    # move the model on the GPU
    model = model.cuda()

    # get the data loader
    data_loader = DataLoader(
        batch_size=16,
        data_path="../data/semeval/train_all_tasks.csv",
        shuffle=False,
        task_column_name=None,
    )

    embeddings = []
    rewire_id = []
    target = []

    for batch in tqdm(data_loader):
        # get input
        inputs = list(batch["text"].values)
        inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        inputs.to("cuda")

        # use the model
        outputs = model(**inputs)[0]
        embeddings.append(outputs.detach().cpu().numpy())

        # get rewire id
        rewire_id.extend(list(batch["rewire_id"].values))

        # get the target
        target.extend(list(batch["label_sexist"].values))

    embeddings = np.concatenate(embeddings)

    # save the embeddings
    np.save("../data/custom/embeddings.npy", embeddings)

    # save the rewire_id
    np.save("../data/custom/rewire_id.npy", np.array(rewire_id))

    # save the target
    np.save("../data/custom/target.npy", np.array(target))


def auto_train(
        data: pd.DataFrame,
        target_column: str,
        use_gpu: bool = False,
        save_path: Optional[str] = None
):
    setup(data=data, target=target_column, session_id=123, use_gpu=use_gpu)
    best_model = compare_models()

    if save_path:
        save_model(best_model, save_path)

    return best_model


def auto_predict(
        data: pd.DataFrame,
        model
):
    unseen_predictions = predict_model(model, data=data)
    return unseen_predictions


def get_train_test():
    # get the embeddings
    embeddings = np.load("../data/custom/embeddings.npy")
    targets = np.load("../data/custom/target.npy")

    # create a dataframe with the embeddings and the targets
    df = pd.DataFrame(embeddings)
    df["target"] = targets

    # split the data in train and test
    train = df.sample(frac=0.95, random_state=42)
    test = df.drop(train.index)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, test


def train_auto_ml():
    # get data
    train, test = get_train_test()

    # create an auto ml model
    model = auto_train(train, 'target', use_gpu=True, save_path="../trained_agents/auto_ml")


def test_auto_ml():
    # get data
    train, test = get_train_test()

    # load the model
    model = load_model("../trained_agents/auto_ml")

    # predict on the test set
    predictions = auto_predict(test, model)

    # Compute accuracy between "target" and "Label"
    accuracy = (predictions["target"] == predictions["Label"]).mean()

    print(accuracy)


if __name__ == "__main__":
    # create_embeddings()
    test_auto_ml()
