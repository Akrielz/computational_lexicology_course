from typing import Callable, Optional

import torch
from einops import rearrange
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm

from project.models.sexist_bert import SexistBert
from project.pipeline.data_loader import DataLoader


def build_model(model_path: Optional[str] = None) -> nn.Module:
    # create model
    model = SexistBert(device="cuda", num_classes=1, pool_method="bert", depth=2).cuda()

    # load the model
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # return the model
    return model


def do_one_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    # set model to train mode
    model.train()

    # init binary accuracy metric
    binary_accuracy = BinaryAccuracy().to("cuda")

    # progress_bar
    progress_bar = tqdm(data_loader)

    # iterate the data loader
    for batch in data_loader:
        # get targets
        targets = batch["label_sexist"].values
        targets = torch.tensor([1 if target == "sexist" else 0 for target in targets])
        targets = targets.to("cuda")
        targets = targets.float()

        # get input
        inputs = list(batch["text"].values)

        # use the model
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
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
    torch.save(model.state_dict(), "../trained_agents/sexist_bert_a.pt")


def train(num_epochs: int):
    # get the data loader
    data_loader = DataLoader(
        batch_size=16,
        shuffle=True,
        seed=42,
        balance_data_method="duplication",
        task_column_name="label_sexist",
    )

    # create model
    model = SexistBert(device="cuda", num_classes=1, pool_method="bert", depth=2).cuda()

    # get the loss function
    loss_function = nn.BCELoss()

    # get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # create lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # iterate the epochs
    for epoch in range(num_epochs):
        do_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            loss_function=loss_function,
        )
        lr_scheduler.step()


if __name__ == "__main__":
    train(num_epochs=1)