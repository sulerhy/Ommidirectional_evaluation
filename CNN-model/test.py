"""
Copyright (C) Hoang Pham Duc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Hoang Pham Duc <phamduchoangeee@gmail.com>, May 2020
"""

import h5py
import numpy as np
import sys
import scipy.io as spio
import utils
np.set_printoptions(threshold=sys.maxsize)

# mat = spio.loadmat('reference_research/WVPSNR.mat', squeeze_me=True)
# X = mat['value']
# print(X)
X_train, y_train, X_test, y_test = utils.load_all_data()
print(X_train[3,:,:,1])
