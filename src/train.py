import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

from model import AnimalClassifier
from load import LOAD

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

def train(model, X, Y, TEST_IMAGES , TEST_LABELS , optimizer, loss_fn, batch_size=BATCH_SIZE, epochs=EPOCHS) :
    model.train()
    train_loss = []
    for epoch in range(epochs) :
        LOSS = []
        for i in range(0, len(X), batch_size) :
            x_batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            y_batch = torch.tensor(Y[i:i+batch_size], dtype=torch.long).to(device)
            # print(x_batch.shape)
            x_batch = x_batch.permute(0, 3, 1, 2)
            # print(x_batch.shape)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            print(f'\rEpoch {epoch} [{i}/{len(X)}] loss {loss.item()} \t\t', end='')
            LOSS.append(loss.item())
        print(f'\rEpoch', epoch, 'loss', np.mean(LOSS))
        train_loss.append(np.mean(LOSS))

    test_loss = []
    model.eval()
    for i in range(0, len(TEST_IMAGES), batch_size) :
        x_batch = torch.tensor(TEST_IMAGES[i:i+batch_size], dtype=torch.float32).to(device)
        y_batch = torch.tensor(TEST_LABELS[i:i+batch_size], dtype=torch.long).to(device)
        x_batch = x_batch.permute(0, 3, 1, 2)
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        test_loss.append(loss.item())
    print('Test loss', np.mean(test_loss))

    return model, train_loss, test_loss

TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS = LOAD()

model, train_loss , test_loss = train(model, TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES , TEST_LABELS , optimizer, loss_fn)

torch.save(model.state_dict(), MODEL_NAME)

plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(['train loss', 'test loss'])
plt.show()