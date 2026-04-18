"""
Module to load any torchvision dataset
"""

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class DataManager:
    def __init__(
        self,
        dataset: torchvision.datasets,
        augment_training: bool = False,
        batch_size: int = 32,
    ):
        self._train_transform = transforms.ToTensor()
        self._test_transform = transforms.ToTensor()
        if augment_training:
            self._augment_training()
        self._load_data(dataset=dataset, batch_size=batch_size)

    def _load_data(self, dataset: torchvision.datasets, batch_size: int = 32):
        self.train_dataset = dataset(
            root="./data", train=True, download=True, transform=self._train_transform
        )
        self.test_dataset = dataset(
            root="./data", train=False, download=True, transform=self._test_transform
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

    def _augment_training(self) -> bool:
        self._train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ]
        )
