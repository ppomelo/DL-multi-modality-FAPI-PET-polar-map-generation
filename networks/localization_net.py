import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(".")
from utils.module import AntiAliasInterpolation3d,rotation_matrix

def calculate_output_size(input_size, kernel_size, stride, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1      

def calculate_final_flattened_size():
    # Initial size
    depth, height, width = 127, 172, 172
    
    # After conv1 and max_pool3d
    depth = calculate_output_size(depth, 3, 1)
    height = calculate_output_size(height, 3, 1)
    width = calculate_output_size(width, 3, 1)
    depth, height, width = calculate_output_size(depth, 1, 1), calculate_output_size(height, 2, 2), calculate_output_size(width, 2, 2)

    # After conv2 and max_pool3d
    depth = calculate_output_size(depth, 3, 1)
    height = calculate_output_size(height, 3, 1)
    width = calculate_output_size(width, 3, 1)
    depth, height, width = calculate_output_size(depth, 1, 1), calculate_output_size(height, 2, 2), calculate_output_size(width, 2, 2)

    # After conv3 and max_pool3d
    depth = calculate_output_size(depth, 3, 1)
    height = calculate_output_size(height, 3, 1)
    width = calculate_output_size(width, 3, 1)
    depth, height, width = calculate_output_size(depth, 1, 1), calculate_output_size(height, 2, 2), calculate_output_size(width, 2, 2)

    # After conv4 and max_pool3d
    depth = calculate_output_size(depth, 3, 1)
    height = calculate_output_size(height, 3, 1)
    width = calculate_output_size(width, 3, 1)
    depth, height, width = calculate_output_size(depth, 1, 1), calculate_output_size(height, 2, 2), calculate_output_size(width, 2, 2)

    return depth * height * width * 16

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.down = AntiAliasInterpolation3d(1, 1)
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=3)
        self.bn3 = nn.BatchNorm3d(16)
        self.conv4 = nn.Conv3d(16, 16, kernel_size=3)
        self.bn4 = nn.BatchNorm3d(16)
    
    def forward(self, x):
        x = self.down(x)
        x = F.max_pool3d(self.bn1(self.conv1(x)), kernel_size=(2, 2, 1), stride=(2, 2, 1))
        x = F.max_pool3d(self.bn2(self.conv2(x)), kernel_size=(2, 2, 1), stride=(2, 2, 1))
        x = F.max_pool3d(self.bn3(self.conv3(x)), kernel_size=(2, 2, 1), stride=(2, 2, 1))
        x = F.max_pool3d(self.bn4(self.conv4(x)), kernel_size=(2, 2, 1), stride=(2, 2, 1))

        return x

class NetworkAngles(nn.Module):
    def __init__(self, custom_bias=0.1):
        super(NetworkAngles, self).__init__()
        self.encoder = Encoder()
        
        # Calculate the flattened size after the encoder
        self.flattened_size = 121856
        # print(self.flattened_size)
        # Define fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)
        # Initialize the weights and biases of the final layer
        nn.init.constant_(self.fc4.weight, 0)
        self.fc4.bias.data = torch.tensor([44, 29, -81], dtype=torch.float32)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # Apply fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x

class NetworkTranslation(nn.Module):
    def __init__(self, custom_bias=0.1):
        super(NetworkTranslation, self).__init__()
        self.encoder = Encoder()
        
        # Calculate the flattened size after the encoder
        self.flattened_size = 121856  # Adjust as necessary
        
        # Define fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)
        
        # Initialize the weights and biases of the final layer
        nn.init.constant_(self.fc4.weight, 0)
        self.fc4.bias.data = torch.tensor([98, 44, 143], dtype=torch.float32)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x



