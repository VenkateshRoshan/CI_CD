import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import AnimalClassifier
from load import LOAD
import unittest
from train import train

BATCH_SIZE = 32
EPOCHS = 10
lr = 0.001

MODEL_NAME = 'models/model.pth'

class TestTrainingProcess(unittest.TestCase) :

    def setup(self) :
        self.model = AnimalClassifier(Y=3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.input_tensor = torch.tensor(LOAD()[0][0:32], dtype=torch.float32).to(self.device)
        self.label_tensor = torch.tensor(LOAD()[1][0:32], dtype=torch.long).to(self.device)

    def test_train(self) :
        self.setup()
        model, train_loss , test_loss = train(self.model, self.input_tensor, self.label_tensor, self.input_tensor, self.label_tensor, self.optimizer, self.loss_fn)
        self.assertTrue(isinstance(model, AnimalClassifier))
        self.assertTrue(isinstance(train_loss, list))
        self.assertTrue(isinstance(test_loss, list))
        self.assertTrue(isinstance(train_loss[0], np.float64))
        self.assertTrue(isinstance(test_loss[0], np.float64))
        self.assertTrue(len(train_loss) == EPOCHS)
        self.assertTrue(len(test_loss) == EPOCHS)

    def testGradientProcess(self) :
        self.setup()
        self.optimizer.zero_grad()
        y_pred = self.model(self.input_tensor)
        loss = self.loss_fn(y_pred, self.label_tensor)
        loss.backward()
        for param in self.model.parameters() :
            self.assertTrue(param.grad is not None)

if __name__ == '__main__' :
    unittest.main()