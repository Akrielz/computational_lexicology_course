from typing import Optional

import numpy as np
import pandas as pd
import torch
from pycaret.classification import compare_models, save_model, setup, predict_model, load_model
from tqdm import tqdm

from project.pipeline.classic_tokenizers import get_bert_tokenizer
from project.pipeline.data_loader import DataLoader
from project.task_a.train_sexist_bert_pretrained_dual import build_model


def auto_predict(
        data: pd.DataFrame,
        model
):
    unseen_predictions = predict_model(model, data=data)
    return unseen_predictions


def create_submission():
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
        data_path="../data/semeval/dev_task_a_entries.csv",
        shuffle=False,
        task_column_name=None,
    )

    embeddings = []
    rewire_id = []

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

    # concatenate the embeddings
    embeddings = np.concatenate(embeddings)

    # create a datafarme with embeddings and rewire_id
    df = pd.DataFrame(embeddings)
    df["rewire_id"] = rewire_id

    # load the model
    model = load_model("../trained_agents/auto_ml")

    # predict on the test set
    predictions = auto_predict(df, model)

    # create a new dataframe with just the rewire_id and the predictions
    df = pd.DataFrame({"rewire_id": predictions["rewire_id"], "label_pred": predictions["Label"]})

    # save the predictions
    df.to_csv("../data/semeval/submission_task_a/dev/best/sexist_auto_ml.csv", index=False)


if __name__ == "__main__":
    create_submission()