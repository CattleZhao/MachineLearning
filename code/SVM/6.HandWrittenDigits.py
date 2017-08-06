# coding = utf-8

import numpy as np
from sklearn import svm
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import GridSearchCV
from time import time
import pandas as pd


if __name__ == '__main__':
    print('Load Training File Start...')
    data = pd.read_csv('optdigits.tra', header=None)
    x, y = data[list(range(64))].values, data[64].values
    images = x.reshape(-1, 8, 8)
    y = y.ravel().astype(np.int)
    print(y)
