"""
Copyright (C) Hoang Pham Duc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Hoang Pham Duc <phamduchoangeee@gmail.com>, May 2020
"""

import numpy as np
import sys
import scipy.io as spio
import json
from pathlib import Path
import keras
import matplotlib.pyplot as plt
import h5py
import scipy.stats.stats as stats
from sklearn.metrics import mean_squared_error
from math import sqrt
import utils


# np.set_printoptions(threshold=sys.maxsize)


mat = spio.loadmat('WVPSNR.mat', squeeze_me=True)
y_predict = mat['value'] / 10.0
y_predict = y_predict[308:]
print(y_predict)

# get testing set ground truth
with h5py.File('../../dataset/y_test.h5', 'r') as hf:
    y_test = hf['y_test'][:]
# evaluate:
utils.evaluate(y_test, y_predict)
