import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(transform, limit=1000, shuffle=True):
    """
    Load CIFAR-10 dataset and preprocess the images with a specified limit
    """
    dataset = CIFAR10(root='./image_dataset/', download=True, transform=transform)

    # Limit the dataset to the specified number of samples
    dataset.data = dataset.data[:limit]
    dataset.targets = dataset.targets[:limit]

    if shuffle:
        # Shuffle the dataset
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        dataset.data = dataset.data[indices]
        dataset.targets = np.array(dataset.targets)[indices].tolist()

    data_loader = DataLoader(dataset, batch_size=1)
    return data_loader


def load_pre_trained_model(select_device):
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
    model_set.to(select_device)

    return model_set


def calculate_similarity(features_of_all_images):
    """
    Calculate the similarity scores between images based on the inverse Euclidean distance of their features.
    """
    similarity_scores_list = []
    for i in range(len(features_of_all_images)):
        scores = []
        for j in range(i, len(features_of_all_images)):
            distance = np.linalg.norm(features_of_all_images[i].cpu() - features_of_all_images[j].cpu())
            if distance == 0:
                score = 1
            else:
                score = 1 / distance
            scores.append((j, score))
        similarity_scores_list.append(scores)
    return similarity_scores_list


def rank_normalization(similarity_lists):
    """
    Rank normalization of the similarity scores
    :param similarity_lists:
    :return:
    """

    normalized_similarity_scores = []
    L = len(similarity_lists[0])  # Length of each similarity list

    for i in range(len(similarity_lists)):
        ranks = []
        for j in range(len(similarity_lists[i])):
            rank = 2 * L - (similarity_lists[i][j][1] + similarity_lists[j][i][1])
            ranks.append((j, rank))  # Append a tuple instead of a list
        normalized_similarity_scores.append(sorted(ranks))

    return normalized_similarity_scores


def get_the_features_of_the_image(image, model):
    image_tensor = transform_pipeline(image)
    features_var = model(image_tensor.unsqueeze(0).to(device))  # extract features
    features = features_var.data.cpu()  # get the tensor out of the variable and copy it to host memory

    # print("Features of the image: ", features.size())

    return features


def preview_image(image):
    print("Shape of the image: ", image.shape)
    print("Type of the image: ", type(image))
    # Reshape the flattened image to its original shape
    image = unflatten_image(image)
    show_image(image)


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

    # Set the processing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the CIFAR-10 dataset
    data_loader = load_dataset(transform_pipeline)

    # Get the images and labels from the dataset
    images = data_loader.dataset.data
    labels = data_loader.dataset.targets

    # Preview the image
    preview_image(images[0])

    # Load the pre-trained model
    pre_trained_model = load_pre_trained_model(device)
    # print(pre_trained_model)

    # Get all the features of the images
    features = []
    for image in images:
        features.append(get_the_features_of_the_image(image, pre_trained_model))

    # Make them 1D
    for i in range(len(features)):
        features[i] = features[i].view(features[i].size(0), -1)

    # Calculate the Euclidean distance between the features of an image and the features of all images and store the
    # similarity scores
    similarity_scores = calculate_similarity(features)

    print(similarity_scores[0])
    print("Length of the euclidean distances: ", len(similarity_scores))

    # Rank normalization of the similarity scores
    normalized_similarity_scores = rank_normalization(similarity_scores)
    print(normalized_similarity_scores[0])
