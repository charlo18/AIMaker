from DataManager import DataManager
import matplotlib.pyplot as plt
import torchvision

if __name__ == "__main__":
    dm = DataManager(dataset=torchvision.datasets.MNIST, augment_training=True)
    images, labels = next(iter(dm.test_loader))

    fig, axes = plt.subplots(5, 5, figsize=(10, 4))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(str(labels[i].item()))
        ax.axis("off")

    plt.show()
