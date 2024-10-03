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
import glob

np.set_printoptions(threshold=sys.maxsize)
model_path = '../../models/convnet_model.json'
weight_path = '../../models/convnet_weights.h5'
max_PSNR = 99.0
n_x = 128.0
n_y = 64.0
# AIZU DATASET
example_num_AIZU = 416
total_ex_AIZU = 512


def get_dataset_AIZU():
    sys.stdout.write('///////////////Loading Dataset AIZU//////////////////\n')
    sys.stdout.flush()
    # list files
    distance_files = glob.glob("../../data/AIZU-dataset/*.mat")
    X_train_AIZU = np.array([])
    y_train_AIZU = np.array([])
    X_test_AIZU = np.array([])
    y_test_AIZU = np.array([])
    for file in distance_files:
        X_train, y_train, X_test, y_test = get_dataset_AIZU_file(file)
        if X_train_AIZU.size == 0:
            X_train_AIZU, y_train_AIZU, X_test_AIZU, y_test_AIZU = X_train, y_train, X_test, y_test
        else:
            X_train_AIZU = np.concatenate((X_train_AIZU, X_train), axis=0)
            y_train_AIZU = np.concatenate((y_train_AIZU, y_train), axis=0)
            X_test_AIZU = np.concatenate((X_test_AIZU, X_test), axis=0)
            y_test_AIZU = np.concatenate((y_test_AIZU, y_test), axis=0)

    print("----AIZU DATASET SHAPE")
    print("X_train shape:" + str(np.shape(X_train_AIZU)))
    print("Y_train shape:" + str(np.shape(y_train_AIZU)))
    print("X_test shape:" + str(np.shape(X_test_AIZU)))
    print("Y_test shape:" + str(np.shape(y_test_AIZU)))

    return X_train_AIZU, y_train_AIZU, X_test_AIZU, y_test_AIZU


def get_dataset_AIZU_file(file):
    mat = spio.loadmat(file, squeeze_me=True)
    X_data = mat['result']

    # There are a lot of inf PSNR value, we need to replace them with maximum value = 99
    X_data[:, :, :, 0] = np.nan_to_num(X_data[:, :, :, 0], posinf=max_PSNR)
    y_data = mat['MOS']
    # rescale test_set 0->1
    X_data[:, :, :, 0] = X_data[:, :, :, 0] / max_PSNR
    # split data into data train and data set
    X_train = X_data[0:example_num_AIZU, :, :, :]
    y_train = y_data[0:example_num_AIZU]
    X_test = X_data[example_num_AIZU:, :, :, :]
    y_test = y_data[example_num_AIZU:]
    # reshape into (example_num, 1)
    y_train = y_train.reshape((example_num_AIZU, 1))
    y_test = y_test.reshape((total_ex_AIZU - example_num_AIZU, 1))
    # double data set by flip horizontally PSNR
    X_flip = np.flip(X_train, axis=2)
    X_train = np.concatenate((X_train, X_flip), axis=0)
    y_flip = y_train
    y_train = np.concatenate((y_train, y_flip), axis=0)
    X_flip = np.flip(X_test, axis=2)
    X_test = np.concatenate((X_test, X_flip), axis=0)
    y_flip = y_test
    y_test = np.concatenate((y_test, y_flip), axis=0)

    return X_train, y_train, X_test, y_test


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


def main():
    # part 1: Load all data:
    X_train, y_train, X_test, y_test = get_dataset_AIZU()

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
