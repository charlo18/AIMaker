"""
Module for training and testing model from torch
"""

# local imports
from Utils.names import names
import Utils.utils as utils

# extern import
import random
import tqdm
import os

# torch imports
from torch import nn
from torch.utils.data import DataLoader
import torch
from torch import optim


class ModelManager:
    def __init__(self, model_name: str, network: nn.Module):
        if model_name is None:
            model_name = random.choices(names) + str(random.randint(0, 10000))
        self.model_name = model_name
        self.network = network

    def train(
        self,
        epochs: int,
        train_dataset: DataLoader,
        weight_decay: float,
        momentum: float,
        loss_function: nn.Module,
        test_dataset: DataLoader = None,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        lr: float = 0.01,
    ):
        optimizer = utils.build_optimizer(
            self.network, lr, optimizer, momentum=momentum, weight_decay=weight_decay
        )

        self.network.train()  # training mode
        for epoch in range(epochs):
            print(epoch)
            for images, labels in tqdm.tqdm(train_dataset):
                pred = self.network(images)  # forward
                loss = loss_function(pred, labels)  # loss
                optimizer.zero_grad()
                loss.backward()  # backward propagation
                optimizer.step()

            # test model
            preds = []
            lbls = []

            for images, labels in test_dataset:
                preds.append(self.inference(images))
                lbls.append(labels)

            preds = torch.cat(preds)
            lbls = torch.cat(lbls)

            acc = self.accuracy(preds, lbls)
            print(f"{acc * 100:.2f}%")

        print("training done")
        self._save_model(epochs, optimizer, train_dataset.dataset.__class__.__name__)

    def inference(self, x):
        self.network.eval()

        with torch.no_grad():
            pred = self.network(x)
            pred = torch.argmax(pred, dim=1)

        return pred

    def _save_model(self, epochs: int, optimizer: optim.Optimizer, dataset_name: str):
        print("Saving model")
        path = os.path.join("Results", self.network.__class__.__name__, self.model_name)
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs": epochs,
                "dataset_name": dataset_name,
            },
            os.path.join(path, self.model_name),
        )
        print("Saving done")

    def accuracy(self, pred, ground):
        correct = (pred == ground).sum().item()
        total = ground.size(0)
        return correct / total
