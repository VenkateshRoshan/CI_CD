import torch
import torch.nn as nn
import unittest

from model import AnimalClassifier

class TestModel(unittest.TestCase) :
    def test_model(self) :
        model = AnimalClassifier()
        self.assertTrue(isinstance(model, nn.Module))
        self.assertTrue(isinstance(model.conv1, nn.Conv2d))
        self.assertTrue(isinstance(model.conv2, nn.Conv2d))
        self.assertTrue(isinstance(model.conv3, nn.Conv2d))
        self.assertTrue(isinstance(model.fc1, nn.Linear))
        self.assertTrue(isinstance(model.fc2, nn.Linear))
        self.assertTrue(isinstance(model.pool, nn.MaxPool2d))
        self.assertTrue(isinstance(model.dropout, nn.Dropout))

    def test_forward(self) :
        model = AnimalClassifier()
        x = torch.randn(2, 3, 128, 128)
        y = model(x)
        self.assertEqual(y.shape, torch.Size([2, 3]))

if __name__ == '__main__' :
    unittest.main()