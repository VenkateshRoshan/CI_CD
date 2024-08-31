import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

class AnimalClassifier(nn.Module) :
    def __init__(self, Y = 3) :
        super(AnimalClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, Y)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x) :
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, 128 * 14 * 14)
        x = nn.Flatten()(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = AnimalClassifier()
# summary(model, (3, 128, 128))