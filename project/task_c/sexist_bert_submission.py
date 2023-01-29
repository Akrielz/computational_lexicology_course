import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from project.pipeline.classic_tokenizers import get_bert_tokenizer
from project.pipeline.data_loader import DataLoader
from project.task_c.sexist_bert_train import build_model


def prediction_to_labels(predictions: np.array):
    """
    category and frequency

    {'1.1 threats of harm': 56,
     '1.2 incitement and encouragement of harm': 254,
     '2.1 descriptive attacks': 717,
     '2.2 aggressive and emotive attacks': 673,
     '2.3 dehumanising attacks & overt sexual objectification': 200,
     '3.1 casual use of gendered slurs, profanities, and insults': 637,
     '3.2 immutable gender differences and gender stereotypes': 417,
     '3.3 backhanded gendered compliments': 64,
     '3.4 condescending explanations or unwelcome advice': 47,
     '4.1 supporting mistreatment of individual women': 75,
     '4.2 supporting systemic discrimination against women as a group': 258,
     'none': 10602}

    We need to convert the category into labels from 0 to 11
    """

    category_map = {
        '1.1 threats of harm': 0,
        '1.2 incitement and encouragement of harm': 1,
        '2.1 descriptive attacks': 2,
        '2.2 aggressive and emotive attacks': 3,
        '2.3 dehumanising attacks & overt sexual objectification': 4,
        '3.1 casual use of gendered slurs, profanities, and insults': 5,
        '3.2 immutable gender differences and gender stereotypes': 6,
        '3.3 backhanded gendered compliments': 7,
        '3.4 condescending explanations or unwelcome advice': 8,
        '4.1 supporting mistreatment of individual women': 9,
        '4.2 supporting systemic discrimination against women as a group': 10,
    }

    inverse_category_map = {v: k for k, v in category_map.items()}

    return [inverse_category_map[prediction] for prediction in predictions]


def create_submission():
    # get tokenizer
    tokenizer = get_bert_tokenizer()

    epoch = 0
    mode = "test"

    # get model
    model = build_model(model_path=f"../trained_agents/sexist_bert_pretrained_c_e_{epoch}.pt").cuda()
    model.eval()

    # get the data loader
    data_loader = DataLoader(
        batch_size=16,
        data_path=f"../data/semeval/{mode}_task_c_entries.csv",
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
    df.to_csv(f"../data/semeval/submission_task_c/{mode}/sexist_bert_e_{epoch}.csv", index=False)


if __name__ == "__main__":
    create_submission()
