import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from project.pipeline.classic_tokenizers import get_bert_tokenizer
from project.pipeline.data_loader import DataLoader
from project.task_b.sexist_bert_train import build_model


def prediction_to_labels(predictions: np.array):
    """
    The categories are:

    {'1. threats, plans to harm and incitement': 310,
     '2. derogation': 1590,
     '3. animosity': 1165,
     '4. prejudiced discussions': 333,
     'none': 10602}
    """

    category_map = {
        '1. threats, plans to harm and incitement': 0,
        '2. derogation': 1,
        '3. animosity': 2,
        '4. prejudiced discussions': 3
    }

    inverse_category_map = {v: k for k, v in category_map.items()}

    return [inverse_category_map[prediction] for prediction in predictions]


def create_submission():
    # get tokenizer
    tokenizer = get_bert_tokenizer()

    epoch = 5
    mode = "test"

    # get model
    model = build_model(model_path=f"../trained_agents/sexist_bert_pretrained_b_e_{epoch}.pt").cuda()
    model.eval()

    # get the data loader
    data_loader = DataLoader(
        batch_size=16,
        data_path=f"../data/semeval/{mode}_task_b_entries.csv",
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
    predictions = prediction_to_labels(predictions)

    # create dataframe with rewire_id and predictions
    df = pd.DataFrame({"rewire_id": rewire_id, "label_pred": predictions})

    # save the dataframe
    df.to_csv(f"../data/semeval/submission_task_b/{mode}/sexist_bert_e_{epoch}.csv", index=False)


if __name__ == "__main__":
    create_submission()