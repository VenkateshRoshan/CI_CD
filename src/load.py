import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

DATA_DIR = 'data/animals/'

LABELS = ['cat', 'dog' , 'horse']

def load_data(PATH) :
    IMAGES, Y = [], []
    
    for label in LABELS :
        print('Loading images for', label)
        for file in os.listdir(PATH + label) :
            image = cv2.imread(PATH + label + '/' + file)
            image = cv2.resize(image, (128, 128))
            IMAGES.append(image)
            Y.append(LABELS.index(label))
        
    IMAGES = np.array(IMAGES).astype('float32') / 255
    Y = np.array(Y)

    return IMAGES , Y

def LOAD() :
    TRAIN_IMAGES, TRAIN_LABELS = load_data(DATA_DIR + 'train/')
    TEST_IMAGES, TEST_LABELS = load_data(DATA_DIR + 'val/')

    return TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS


