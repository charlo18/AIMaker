from Managers.DataManager import DataManager
from Managers.modelManager import ModelManager
import torchvision
from models.model import RCNN
import torch.nn as nn

if __name__ == "__main__":
    dataManager = DataManager(dataset=torchvision.datasets.MNIST, augment_training=True)
    # get dims for one image
    images, labels = next(iter(dataManager.train_loader))
    in_features = images[0].shape[0]

    model = RCNN(in_features, dataManager.nb_classes)
    modelManager = ModelManager(model_name="test", network=model)
    modelManager.train(
        epochs=10,
        train_dataset=dataManager.train_loader,
        test_dataset=dataManager.test_loader,
        momentum=0.1,
        weight_decay=0.1,
        loss_function=nn.CrossEntropyLoss(),
    )
