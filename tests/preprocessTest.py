import unittest
from load import LOAD

class TestDataLoading(unittest.TestCase):
    def test_data_loading(self):
        TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS = LOAD()
        # Check Images shape is 128,128,3
        self.assertEqual(TRAIN_IMAGES.shape[1], 128)
        self.assertEqual(TRAIN_IMAGES.shape[2], 128)
        self.assertEqual(TRAIN_IMAGES.shape[3], 3)
        
        self.assertEqual(TEST_IMAGES.shape[1], 128)
        self.assertEqual(TEST_IMAGES.shape[2], 128)
        self.assertEqual(TEST_IMAGES.shape[3], 3)


if __name__ == '__main__':
    unittest.main()