from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import pickle


def load_dataset():
    """
        Load CIFAR-10 dataset and preprocess the images
    """
    dataset = CIFAR10(root='./image_dataset/', download=True)

    images = dataset.data
    labels = dataset.targets

    return images, labels

def show_image(image):
    """
        Display the image
    """
    plt.imshow(image)
    plt.show()

def unflatten_image(image):
    """
        Reshape the flattened image to its original shape
    """
    original_shape = (32, 32, 3)
    return image.reshape(original_shape)

if __name__ == "__main__":
    print("Image Analysis: Final Exam")

    transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    images, labels = load_dataset()
    print("Number of images in the dataset: ", len(images))

    print("Shape of the image: ", images[0].shape)
    print("Type of the image: ", type(images[0]))

    # Reshape the flattened image to its original shape
    image = unflatten_image(images[154])
    show_image(image)

