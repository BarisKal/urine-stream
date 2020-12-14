import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ComplexCNN(nn.Module):
    def __init__(self, use_dropout: bool = True):
        super().__init__()

        self.block1 = nn.Sequential(
            ConvLayer(in_channels=3, out_channels=32),
            ConvLayer(in_channels=32, out_channels=32),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(
            ConvLayer(in_channels=32, out_channels=64),
            ConvLayer(in_channels=64, out_channels=64),
            ConvLayer(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block3 = nn.Sequential(
            ConvLayer(in_channels=64, out_channels=128),
            ConvLayer(in_channels=128, out_channels=128),
            ConvLayer(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=8),
        )

        if use_dropout:
            # use dropout layer in fully connected classification part like in ImageNet paper
            self.fc = nn.Sequential(
                nn.Linear(in_features=128, out_features=64),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=2)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_features=128, out_features=64),
                nn.Linear(in_features=64, out_features=2)
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.view(-1, 128)
        yhat = self.fc(x)

        return yhat