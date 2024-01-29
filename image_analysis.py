import random
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Load pre-trained ResNet-18 model
model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
# Change the output size for CIFAR-10 (10 classes)
model.fc = nn.Linear(512, 10)
# Remove the classification layer (the last layer)
model = nn.Sequential(*list(model.children())[:-1])
# Set the model to evaluation mode
model.eval()

# Load and preprocess CIFAR-10 images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = datasets.CIFAR10(root='./image_dataset/', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

print(len(dataset)) # 50000

print("Type of dataset: ", type(dataset)) # <class 'torchvision.datasets.cifar.CIFAR10'>

# Convert the dataset to list
dataset = list(dataset)

# shuffle the dataset
random.shuffle(dataset)

dataset = dataset[:5000]

print(len(dataset)) # 5000

print(dataset[0])

# Get the first image and its label
image, label = dataset[0]
print("Image shape: ", image.shape) # torch.Size([3, 32, 32])
print("Label: ", label) # 6


img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow('Image', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Forward pass to obtain features for all images in the dataset
# with torch.no_grad():
#     all_features = []
#     for images, _ in data_loader:
#         features = model(images)
#         all_features.append(features)
#
# # Flatten the features to make them a 1D tensor
# flat_features = torch.cat(all_features, dim=0).view(len(dataset), -1)
#
# # Now 'flat_features' contains the extracted features from the chosen layer for all CIFAR-10 images
# print(flat_features)