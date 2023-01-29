from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from project.pipeline.classic_tokenizers import get_bert_tokenizer
from project.pipeline.data_loader import DataLoader


def build_model(model_path: Optional[str] = None) -> nn.Module:
    # create model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=12)

    # load the model
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # return the model
    return model


def get_category(targets: np.ndarray) -> torch.Tensor:
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

    targets = [category_map[category] for category in targets]
    targets = torch.tensor(targets)

    return targets


def do_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        tokenizer,
        epoch_num: int,
        save_model: bool = True,
        task_column_name: str = "label_sexist",
):
    # set model to train mode
    model.train()
    model.to("cuda")

    # init binary accuracy metric
    accuracy = Accuracy().to("cuda")

    # init f1 macro metric
    f1_macro = F1Score(average="macro", num_classes=5).to("cuda")

    # progress_bar
    progress_bar = tqdm(data_loader)

    # iterate the data loader
    for batch in data_loader:
        # get targets
        targets = batch[task_column_name].values
        targets = get_category(targets)
        targets = targets.to("cuda")
        targets = targets.long()

        # get input
        inputs = list(batch["text"].values)
        inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        inputs.to("cuda")

        # use the model
        outputs = model(**inputs)[0]

        # get the loss
        loss = loss_function(outputs, targets)

        # zero the gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update the weights
        optimizer.step()

        # add to progress bar the accuracy and loss

        # apply the argmax
        output_labels = torch.argmax(outputs, dim=1)

        accuracy.update(output_labels, targets)
        f1_macro.update(output_labels, targets)
        progress_bar.set_postfix(
            loss=loss.item(),
            accuracy=accuracy.compute().item(),
            f1=f1_macro.compute().item()
        )

        progress_bar.update(1)

    if save_model:
        torch.save(model.state_dict(), f"../trained_agents/sexist_bert_pretrained_b_e_{epoch_num}.pt")


def train(num_epochs: int):
    task_column_name = "label_vector"

    # get the data loader
    data_loader = DataLoader(
        data_path=[
            "../data/semeval/train_all_tasks.csv",
        ],
        batch_size=16,
        shuffle=True,
        seed=42,
        balance_data_method="duplication",
        task_column_name=task_column_name,
    )

    # create model
    model = build_model()

    # get tokenizer
    tokenizer = get_bert_tokenizer()

    # get the loss function
    loss_function = nn.CrossEntropyLoss()

    # get the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # create lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # iterate the epochs
    for epoch in range(num_epochs):
        do_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            tokenizer=tokenizer,
            epoch_num=epoch,
            save_model=True,
            task_column_name=task_column_name,
        )
        lr_scheduler.step()


if __name__ == "__main__":
    train(num_epochs=2)
