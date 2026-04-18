import torch.optim as optim
from typing import Type
from torch.nn import Module


def build_optimizer(
    model: Module,
    lr: float = 0.01,
    optimizer: Type[optim.Optimizer] = optim.SGD,
    **kwargs
) -> optim.Optimizer:
    return optimizer(model.parameters(), lr=lr, **kwargs)
