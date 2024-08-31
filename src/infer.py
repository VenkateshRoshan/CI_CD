import numpy as np
import matplotlib.pyplot as plt

import torch
import cv2

from model import AnimalClassifier
from load import LABELS
from train import MODEL_NAME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AnimalClassifier(Y=3)
model.load_state_dict(torch.load(MODEL_NAME))


INFER_PATH = 'data/animals/inf/'

def inferImg(path) :
    IMG = []
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    IMG.append(img)

    model.eval()
    with torch.no_grad() :
        x = torch.tensor(IMG, dtype=torch.float32).to(device)
        x = x.permute(0, 3, 1, 2)
        y_pred = model(x)
        print(y_pred)
        print(LABELS[torch.argmax(y_pred).item()])
        plt.imshow(IMG[0])
        plt.show()
    