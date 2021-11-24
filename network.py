import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 1 * 25, out_features=1000),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(nn.Linear(in_features=500, out_features=4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
