import numpy as np
import torch
from src.model import AnimalClassifier
from src.load import LOAD
from src.train import train
import torch.optim as optim
import unittest

def accuracy(y_pred, y_true) :
    with torch.no_grad() :
        y_pred = torch.argmax(y_pred, dim=1)
        correct = (y_pred == y_true).sum().item()
        return correct / len(y_true)
    
class TestUtilityFunctions(unittest.TestCase):

    def setUp(self):
        self.output = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
        self.target = torch.tensor([1, 0])
    
    def test_accuracy(self):
        acc = accuracy(self.output, self.target)
        self.assertEqual(acc, 1.0)

if __name__ == '__main__':
    unittest.main()