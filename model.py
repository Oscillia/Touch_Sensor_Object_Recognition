import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms as T


class Net(nn.Module):

    def __init__(self, out_features=5, re_size=32, dropout=0.1):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # re_size/2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # re_size/4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # re_size/8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # re_size/16
            nn.Conv2d(in_channels=256, out_channels=468, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(468),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(in_channels=468, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # re_size/32
            Flatten(),
            nn.Linear(in_features=512 * (re_size // 32) ** 2, out_features=out_features),
        )
        self.re_size = re_size

    def forward(self, x):
        x = self.pre_process(x)
        x = self.model(T.Resize(self.re_size)(x))
        return x

    def pre_process(self, x):
        return x.float()


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        return x.view(batch_size, -1)
