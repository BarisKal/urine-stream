import torch.nn as nn

class ComplexCNN(nn.Module):
    """
    Definition of a more compex (deep) CNN than defined in basiccnn_model.py
    """
    def __init__(self, use_dropout: bool = True, dropout_probability: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            ConvLayer(in_channels=3, out_channels=96),
            ConvLayer(in_channels=96, out_channels=96),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            ConvLayer(in_channels=96, out_channels=192),
            ConvLayer(in_channels=192, out_channels=192),
            ConvLayer(in_channels=192, out_channels=192),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            ConvLayer(in_channels=192, out_channels=384),
            ConvLayer(in_channels=384, out_channels=384),
            ConvLayer(in_channels=384, out_channels=384),
            nn.MaxPool2d(kernel_size=8),
        )
        if use_dropout:
            # use dropout layer in fully connected classification part like in ImageNet paper
            self.fc = nn.Sequential(
                nn.Linear(in_features=3*3*384, out_features=192),
                nn.Dropout(dropout_probability),
                nn.Linear(in_features=192, out_features=2)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_features=3*3*384, out_features=192),
                nn.Linear(in_features=192, out_features=2)
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        #print(x.shape)
        x = x.view(x.size(0), 3*3*384)
        yhat = self.fc(x)

        return yhat

class ConvLayer(nn.Module):
    """
    Serves as encapsulation of ComplexCNN
    """
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        #self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        x = self.relu(x)
        return x