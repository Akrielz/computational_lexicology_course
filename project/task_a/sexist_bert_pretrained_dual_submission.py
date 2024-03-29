import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from project.pipeline.classic_tokenizers import get_bert_tokenizer
from project.pipeline.data_loader import DataLoader
from project.task_a.train_sexist_bert_pretrained_dual import build_model


def create_submission():
    # get tokenizer
    tokenizer = get_bert_tokenizer()

    # get model
    model = build_model(model_path="../trained_agents/sexist_bert_pretrained_dual_a_original.pt").cuda()
    model.eval()

    # get the data loader
    data_loader = DataLoader(
        batch_size=16,
        data_path="../data/semeval/test_task_a_entries.csv",
        shuffle=False,
        task_column_name=None,
    )

    predictions = []
    rewire_id = []

    for batch in tqdm(data_loader):
        # get input
        inputs = list(batch["text"].values)
        inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        inputs.to("cuda")

        # use the model
        outputs = model(**inputs)[0]
        outputs = torch.argmax(outputs, dim=1)
        predictions.append(outputs.detach().cpu().numpy())

        # get rewire id
        rewire_id.extend(list(batch["rewire_id"].values))

    predictions = np.concatenate(predictions)
    predictions = ["sexist" if prediction == 1 else "not sexist" for prediction in predictions]

    # create dataframe with rewire_id and predictions
    df = pd.DataFrame({"rewire_id": rewire_id, "label_pred": predictions})

    # save the dataframe
    df.to_csv("../data/semeval/submission_task_a/test/dual_sexist_bert.csv", index=False)


if __name__ == "__main__":
    create_submission()