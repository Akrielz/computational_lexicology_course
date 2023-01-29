import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from project.pipeline.classic_tokenizers import get_bert_tokenizer
from project.pipeline.data_loader import DataLoader
from project.task_a.train_sexist_bert_pretrained_dual import build_model


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
        'none': 0,
        '1.1 threats of harm': 1,
        '1.2 incitement and encouragement of harm': 2,
        '2.1 descriptive attacks': 3,
        '2.2 aggressive and emotive attacks': 4,
        '2.3 dehumanising attacks & overt sexual objectification': 5,
        '3.1 casual use of gendered slurs, profanities, and insults': 6,
        '3.2 immutable gender differences and gender stereotypes': 7,
        '3.3 backhanded gendered compliments': 8,
        '3.4 condescending explanations or unwelcome advice': 9,
        '4.1 supporting mistreatment of individual women': 10,
        '4.2 supporting systemic discrimination against women as a group': 11,
    }

    inverse_category_map = {v: k for k, v in category_map.items()}

    return [inverse_category_map[prediction] for prediction in predictions]


def create_submission():
    # get tokenizer
    tokenizer = get_bert_tokenizer()

    # get model
    model = build_model(model_path="../trained_agents/sexist_bert_pretrained_c_e_0.pt").cuda()
    model.eval()

    # get the data loader
    data_loader = DataLoader(
        batch_size=16,
        data_path="../data/semeval/test_task_c_entries.csv",
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
    df.to_csv("../data/semeval/submission_task_c/test/sexist_bert_e_0.csv", index=False)


if __name__ == "__main__":
    create_submission()
