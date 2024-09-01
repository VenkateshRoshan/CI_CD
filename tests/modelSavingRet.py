import os
import unittest
from src.model import AnimalClassifier

class TestModelSaveRet(unittest.TestCase) :

    def setup(self) :
        self.model = AnimalClassifier(Y=3)

    def test_model_save(self) :
        os.system('python src/modelSavingRet.py')
        self.assertTrue(os.path.exists('models/model.pth'))

    def test_model_return(self) :
        os.system('python src/modelSavingRet.py')
        self.assertTrue(os.path.exists('models/model.pth'))