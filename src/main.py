import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

from model import AnimalClassifier
from load import LOAD
from train import train

BATCH_SIZE = 32
EPOCHS = 10
lr = 0.001

MODEL_NAME = 'models/model.pth'

model = AnimalClassifier(Y=3)

# train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS = LOAD()

model, train_loss , test_loss = train(model, TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES , TEST_LABELS , optimizer, loss_fn)

torch.save(model.state_dict(), MODEL_NAME)

plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(['train loss', 'test loss'])
plt.show()