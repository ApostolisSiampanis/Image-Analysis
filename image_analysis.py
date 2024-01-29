import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def load_dataset(transform):
    """
        Load CIFAR-10 dataset and preprocess the images
    """
    dataset_original = datasets.CIFAR10(root='./image_dataset/', train=True, download=True, transform=transform)
    dataset = DataLoader(dataset_original, batch_size=1, shuffle=True)
    print("Type of dataset: ", type(dataset))  # <class 'torch.utils.data.dataloader.DataLoader'>
    # Convert the dataset to list
    dataset = list(dataset)
    dataset = dataset[:5000]

    # Get the first image and its label
    image, label = dataset[0]
    print("Image shape: ", image.shape)  # torch.Size([3, 32, 32])
    print("Label: ", label)  # 6

    # Display the first image
    img_rgb = image.numpy().transpose((1, 2, 0))
    plt.imshow(img_rgb)
    plt.title(f"Label: {label}")
    plt.show()

    return dataset


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print("Image Analysis")
    dataset = load_dataset(transform)


# # Load pre-trained ResNet-18 model
# model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
# # Change the output size for CIFAR-10 (10 classes)
# model.fc = nn.Linear(512, 10)
# # Remove the classification layer (the last layer)
# model = nn.Sequential(*list(model.children())[:-1])
# # Set the model to evaluation mode
# model.eval()
