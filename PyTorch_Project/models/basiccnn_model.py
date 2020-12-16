import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    """
    This class defines the structure of a very basic CNN.
    """
    def __init__(self):
        super().__init__()
        # Convolutional Layer (sees 32x32x3 image tensor | outputs 16x16x16 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        # Convolutional Layer (sees 4x4x64 image tensor | outputs 2x2x128 image tensor)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        
        # MaxPooling Layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(8*8*128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        #print(x.shape)
        x = x.view(-1, 8*8*128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        yhat = F.relu(self.fc3(x))
        return yhat