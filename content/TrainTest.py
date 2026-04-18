"""
Module for training and testing model from torch
"""

# local imports
from names import names
import utils

# extern import
import random
import tqdm

# torch imports
from torch import nn
from torch.utils.data import DataLoader
import torch


class TrainTestManager:
    def __init__(self, model_name: str, network: nn.module):
        if model_name is None:
            model_name = random.choices(names) + str(random.randint(0, 10000))
        self.model_name = model_name
        self.network = network

    def train(
        self,
        epochs: int,
        train_dataset: DataLoader,
        test_dataset: DataLoader,
        weight_decay: float,
        momentum: float,
        loss_function: nn.Module,
        optimizer: torch.optim = torch.optim.SGD,
        lr: float = 0.01,
    ):
        optim = utils.build_optimizer(
            self.network, lr, optimizer, momentum, weight_decay
        )
        self.network.train()  # training mode
        for epoch in range(epochs):
            print(epoch)
            for images, labels in tqdm.tqdm(train_dataset):
                pred = self.network(images)  # forward
                loss = loss_function(pred, labels)  # loss
                optim.zero_grad()
                loss.backward()  # backward propagation
                optim.step()

    def test(self): ...
