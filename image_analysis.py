import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt


def load_dataset(transforms):
    """
        Load CIFAR-10 dataset and preprocess the images
    """
    dataset = CIFAR10(root='./image_dataset/', download=True, transform=transforms)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    return data_loader


def pre_trained_model():
    """
        Load the pre-trained model
    """
    model_set = models.resnet18(weights='IMAGENET1K_V1')

    # Set the model to evaluation mode to avoid updating the running statistics of batch normalization layers
    model_set.eval()

    # Freeze the parameters
    for param in model_set.parameters():
        param.requires_grad = False

    # # Replace the last fully-connected layer
    # model_set.fc = nn.Linear(model_set.fc.in_features, 10)  # 10 classes in CIFAR-10

    # Remove the classification (fully connected) layer
    model_set = torch.nn.Sequential(*(list(model_set.children())[:-1]))

    # set processing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_set.to(device)

    return model_set


def show_image(display_image):
    """
        Display the image
    """
    plt.imshow(display_image)
    plt.show()


def unflatten_image(flattened_image):
    """
        Reshape the flattened image to its original shape
    """
    original_shape = (32, 32, 3)
    return flattened_image.reshape(original_shape)


if __name__ == "__main__":
    print("Image Analysis: Final Exam")

    transform_pipeline = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the CIFAR-10 dataset
    data_loader = load_dataset(transform_pipeline)

    # Get the images and labels from the dataset
    images = data_loader.dataset.data
    labels = data_loader.dataset.targets

    print("Number of images in the dataset: ", len(images))

    print("Shape of the image: ", images[0].shape)
    print("Type of the image: ", type(images[0]))
    print(labels[0])

    print("Image transform: ", images[0])

    # Reshape the flattened image to its original shape
    image = unflatten_image(images[0])
    show_image(image)

    # Load the pre-trained model
    model = pre_trained_model()
    print(model)
