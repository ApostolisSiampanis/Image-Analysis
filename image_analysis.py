import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder


def load_dataset(transform, limit=200, shuffle=True):
    """
    Load Stanford Dogs dataset and preprocess the images with a specified limit
    http://vision.stanford.edu/aditya86/ImageNetDogs/
    """
    dataset = ImageFolder('images_dataset', transform=transform)
    if shuffle:
        sample_list_index = random.sample(range(len(dataset)), limit)
        dataset = torch.utils.data.Subset(dataset, sample_list_index)
    else:
        dataset = torch.utils.data.Subset(dataset, range(limit))

    return dataset


def load_pre_trained_model(select_device):
    """
        Load the pre-trained model
    """
    model_set = models.resnet50(weights='IMAGENET1K_V1')

    # Set the model to evaluation mode to avoid updating the running statistics of batch normalization layers
    model_set.eval()

    # Freeze the parameters
    for param in model_set.parameters():
        param.requires_grad = False

    # Remove the classification (fully connected) layer
    model_set = torch.nn.Sequential(*(list(model_set.children())[:-1]))

    # set processing device
    model_set.to(select_device)

    return model_set


def calculate_similarity(features_of_all_images):
    """
    Calculate the similarity scores between images based on the inverse Euclidean distance of their features.
    Similarity score = 1 / (Euclidean distance)
    """
    similarity_scores_list = []
    for i in range(len(features_of_all_images)):
        scores = []
        for j in range(len(features_of_all_images)):
            distance = np.linalg.norm(features_of_all_images[i] - features_of_all_images[j])
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
        # sort the ranks based on the rank (the second value of the tuple) and append to the normalized_similarity_scores list
        normalized_similarity_scores.append(sorted(ranks, key=lambda x: x[1]))

    return normalized_similarity_scores


def get_the_features_of_the_image(image, model):
    image_tensor = transform_pipeline(image)
    features_var = model(image_tensor.unsqueeze(0).to(device))  # extract features
    features = features_var.data.cpu()  # get the tensor out of the variable and copy it to host memory

    # print("Features of the image: ", features.size())

    return features


def get_hypergraph_construction(similarity_scores, k=9):
    hyperedges = []
    for i in range(len(similarity_scores)):
        hyperedge = []
        for j in range(k):
            # If it is, add the node to the hyperedge
            hyperedge.append(similarity_scores[i][j][0])
        # Add the hyperedge to the hyperedges
        hyperedges.append(hyperedge)
    return hyperedges


def create_edge_associations(hyperedges, k=9):
    associations = np.zeros((len(hyperedges), len(hyperedges)))
    for i, e in enumerate(hyperedges):
        for j in range(len(hyperedges)):
            if j in e:
                position = e.index(j) + 1  # Get the position of the node in the hyperedge
                associations[i][j] = 1 - math.log(position, k + 1)  # Calculate the weight
            else:
                associations[i][j] = 0
    return associations


def create_edge_weights(hyperedges, edge_associations):
    weights = []
    for i, e in enumerate(hyperedges):
        sum = 0
        for h in e:
            sum += edge_associations[i][h]
        weights.append(sum)
    return weights


def get_hyperedges_similarities(incidence_matrix):
    Similarity_matrix_h = incidence_matrix @ incidence_matrix.T  # Matrix multiplication
    Similarity_matrix_u = incidence_matrix.T @ incidence_matrix  # Matrix multiplication
    Similarity_matrix = np.multiply(Similarity_matrix_h, Similarity_matrix_u)  # Hadamard product
    return Similarity_matrix


def get_cartesian_product_of_hyperedge_elements(edge_weights, edge_associations, hyperedges):
    membership_degrees = [{} for _ in range(len(hyperedges))]
    matrix_c = np.zeros((len(hyperedges), len(hyperedges)))
    for i, e in enumerate(hyperedges):
        # eq_ei = np.transpose([np.tile(e, len(e)), np.repeat(e, len(e))])
        eq_ei = np.transpose(np.meshgrid(e, e)).reshape(-1, 2)
        for (vertices1, vertices2) in eq_ei:
            membership_degrees[i][(vertices1, vertices2)] = edge_weights[i] * edge_associations[i][vertices1] * \
                                                            edge_associations[i][vertices2]
        for (vertices1, vertices2) in eq_ei:
            matrix_c[vertices1][vertices2] += membership_degrees[i][(vertices1, vertices2)]
    return matrix_c


def get_hypergrapgh_based_simalarity(matrix_c, hyperedges_similarities):
    affinity_matrix = np.multiply(matrix_c, hyperedges_similarities)
    return affinity_matrix


if __name__ == "__main__":
    print("Image Analysis: Final Exam")

    transform_pipeline = transforms.Compose([
        transforms.ToTensor()
    ])

    # Set the processing device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the Stanford Dogs dataset
    data_loader = load_dataset(transform_pipeline)

    # Load the pre-trained model
    pre_trained_model = load_pre_trained_model(device)
    # print(pre_trained_model)

    # Get the first image from the dataset
    first_image, _ = next(iter(data_loader))

    # Convert the image tensor to a numpy array
    first_image = first_image.squeeze().permute(1, 2, 0).numpy()

    # Display the first image
    plt.imshow(first_image)
    plt.axis('off')
    plt.show()

    # Get all the features of the images
    features = []
    for image, _ in data_loader:
        # Add batch dimension to image tensor
        image = image.unsqueeze(0)

        # Move the image tensor to the same device as the pre-trained model
        image = image.to(device)

        # Pass the image through the ResNet18 model
        with torch.no_grad():
            feature = pre_trained_model(image).squeeze().cpu().numpy()  # Convert feature to numpy array

        # Append the feature vector to the list of features
        features.append(feature)

    # Make them 1D
    for i in range(len(features)):
        features[i] = features[i].reshape(features[i].size)

    # ---LHRR Alogrithm---

    # Calculate the Euclidean distance between the features of an image and the features of all images and store the
    # similarity scores
    similarity_scores = calculate_similarity(features)

    print(similarity_scores[0])
    print("Length of the euclidean distances: ", len(similarity_scores))

    # Rank normalization of the similarity scores
    normalized_similarity_scores = rank_normalization(similarity_scores)
    print(normalized_similarity_scores[0])

    # Get the Hyperedges
    hyperedges = get_hypergraph_construction(normalized_similarity_scores)
    print(hyperedges[0])

    edge_associations = create_edge_associations(hyperedges)
    print(edge_associations[0])

    edge_weights = create_edge_weights(hyperedges, edge_associations)
    print(edge_weights[0])

    hyperedges_similarities = get_hyperedges_similarities(edge_associations)
    print(hyperedges_similarities)

    matrix_c = get_cartesian_product_of_hyperedge_elements(edge_weights, edge_associations, hyperedges)

    affinity_matrix = get_hypergrapgh_based_simalarity(matrix_c, hyperedges_similarities)
