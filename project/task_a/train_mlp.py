import pickle
from typing import Callable, Optional

import torch
from einops import rearrange
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
from vision_models_playground.components.activations import GEGLU
from vision_models_playground.components.attention import FeedForward

from project.pipeline.data_loader import DataLoader


def build_model(model_path: Optional[str] = None) -> nn.Module:
    # create model
    model = nn.Sequential(
        FeedForward(dim=6, hidden_dim=64, dropout=0.1, activation=GEGLU(), output_dim=32),
        FeedForward(dim=32, hidden_dim=64, dropout=0.1, activation=GEGLU(), output_dim=32),
        FeedForward(dim=32, hidden_dim=64, dropout=0.1, activation=GEGLU(), output_dim=1),
        nn.Sigmoid(),
    )

    # load the model
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # return the model
    return model


def do_one_epoch(
        features: torch.Tensor,
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device
):
    # set model to train mode
    model.train()

    # move model to device
    model.to(device)

    # init binary accuracy metric
    binary_accuracy = BinaryAccuracy().to(device)

    # progress_bar
    progress_bar = tqdm(data_loader)

    # iterate the data loader
    for batch_indices in data_loader.iter_batch_indices():
        # get targets
        targets = data_loader.df.iloc[batch_indices]["label_sexist"].values
        targets = torch.tensor([1 if target == "sexist" else 0 for target in targets])
        targets = targets.to(device)
        targets = targets.float()

        # get input
        inputs = features[batch_indices]
        inputs = inputs.to(device)

        # use the model
        outputs = model(inputs)
        outputs = rearrange(outputs, "b 1 -> b")

        # get the loss
        loss = loss_function(outputs, targets)

        # zero the gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update the weights
        optimizer.step()

        # add to progress bar the accuracy and loss
        binary_accuracy.update(outputs, targets)
        progress_bar.set_postfix(
            loss=loss.item(),
            accuracy=binary_accuracy.compute().item()
        )

        progress_bar.update(1)

    # save the model
    torch.save(model.state_dict(), "../trained_agents/feed_forward.pt")


def train(num_epochs: int):
    # get the data loader
    data_loader = DataLoader(
        batch_size=32,
        shuffle=True,
        seed=42,
        balance_data_method="duplication",
        task_column_name="label_sexist",
    )

    # read project/task_a/toxic_bert_results.pickle
    with open("../cached/toxic_bert_results.pickle", "rb") as f:
        features = pickle.load(f)
        features = torch.tensor(list(features.values()))
        features = rearrange(features, "f n -> n f")

    # create model
    model = build_model()

    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the loss function
    loss_function = nn.BCELoss()

    # get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # create lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # iterate the epochs
    for epoch in range(num_epochs):
        do_one_epoch(
            features=features,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )
        lr_scheduler.step()


if __name__ == "__main__":
    train(num_epochs=100)