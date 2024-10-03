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
import utils
import init_dataset

np.set_printoptions(threshold=sys.maxsize)
# KAIST DATASET
example_num_KAIST = 576
total_ex_KAIST = 720
model_path = '../../models/convnet_model.json'
weight_path = '../../models/convnet_weights.h5'
max_PSNR = 99.0
n_x = 128.0
n_y = 64.0


def load_model():
    if not Path(model_path).is_file():
        sys.stdout.write('Please train model using basic_model.py first')
        sys.stdout.flush()
        raise SystemExit

    with open(model_path) as file:
        model = keras.models.model_from_json(json.load(file))
        file.close()

    model.load_weights(weight_path)

    return model


def get_dataset_KAIST(distance_file):
    sys.stdout.write('////////////////Loading Dataset KAIST////////////////////\n')
    sys.stdout.flush()
    # part 1: Load all data:
    # MOS
    MOS = spio.loadmat('../../data/MOS.mat', squeeze_me=True)
    y_data = MOS['MOS']
    # distance
    distance = spio.loadmat(distance_file, squeeze_me=True)
    distance = distance['distance']
    # PSNR
    PSNR = spio.loadmat('../../data/PSNR_KAIST.mat', squeeze_me=True)
    PSNR = PSNR['PSNR']
    # There are a lot of inf PSNR value, we need to replace them with maximum value = 99
    PSNR = np.nan_to_num(PSNR, posinf=max_PSNR)

    # combine to test set
    X_data = np.zeros((720, 64, 128, 3))
    X_data[:, :, :, 0] = PSNR
    X_data[:, :, :, 1:3] = distance
    # rescale test_set 0->1
    X_data[:, :, :, 0] = X_data[:, :, :, 0] / max_PSNR
    X_data[:, :, :, 1] = X_data[:, :, :, 1] / n_x
    X_data[:, :, :, 2] = X_data[:, :, :, 2] / n_y
    # split data into data train and data set
    # 5-fold cross validation
    X_test = X_data[init_dataset.start_test:init_dataset.end_test, :, :, :]
    y_test = y_data[init_dataset.start_test:init_dataset.end_test]
    X_train_1 = X_data[0:init_dataset.start_test, :, :, :]
    X_train_2 = X_data[init_dataset.end_test:, :, :, :]
    X_train = np.concatenate((X_train_1, X_train_2), axis=0)
    y_train_1 = y_data[0:init_dataset.start_test]
    y_train_2 = y_data[init_dataset.end_test:]
    y_train = np.concatenate((y_train_1, y_train_2), axis=0)
    # reshape into (example_num, 1)
    y_train = y_train.reshape((np.size(y_train), 1))
    y_test = y_test.reshape((np.size(y_test), 1))
    print("----KAIST DATASET SHAPE")
    print("X_train shape:" + str(np.shape(X_train)))
    print("Y_train shape:" + str(np.shape(y_train)))
    print("X_test shape:" + str(np.shape(X_test)))
    print("Y_test shape:" + str(np.shape(y_test)))

    return X_train, y_train, X_test, y_test


def main():
    # part 1: Load all data:
    X_train, y_train, X_test, y_test = get_dataset_KAIST('../../data/distance/distance_center.mat')
    # step 2: load model
    model = load_model()

    # Part 1: evaluate (training set)
    predict_result = model.predict(X_train)
    utils.evaluate(y_train, predict_result)
    # Part 2: evaluate (test set)
    predict_result = model.predict(X_test)
    utils.evaluate(y_test, predict_result)


if __name__ == "__main__":
    # execute only if run as a script
    main()
