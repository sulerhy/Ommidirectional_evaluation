"""
Copyright (C) Hoang Pham Duc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Hoang Pham Duc <phamduchoangeee@gmail.com>, May 2020
"""

import sys
import json
from pathlib import Path
import numpy as np
import keras
import matplotlib.pyplot as plt
import h5py
import scipy.stats.stats as stats
from sklearn.metrics import mean_squared_error
from math import sqrt
import utils

model_path = './models/convnet_model.json'
weight_path = './models/convnet_weights.h5'
np.set_printoptions(threshold=sys.maxsize)


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


def print_accuracy(model, X_train, y_train, X_test=None, y_test=None):
    print('Calculating model accuracy')

    pred_train = model.predict(X_train)
    acc_train = 1 - np.mean(np.divide(np.absolute(pred_train - y_train), y_train), axis=0)
    print('Training acc: {}'.format(acc_train))

    pred_test = model.predict(X_test)
    acc_test = 1 - np.mean(np.divide(np.absolute(pred_test - y_test), y_test), axis=0)
    print('Testing acc: {}\n'.format(acc_test))

    # evaluate testing:
    utils.evaluate(y_train, pred_train)
    # evaluate testing:
    utils.evaluate(y_test, pred_test)
    sys.stdout.flush()


def print_weight(model):
    print("Final Weight:")
    i = 1
    for layer in model.layers:
        weights = layer.get_weights()  # list of numpy arrays
        print("Layer:" + str(i) + "   ///////////////////////////////////////////////////////////////////")
        print(weights)
        i += 1


def main():
    X_train, y_train, X_test, y_test = utils.load_all_data()
    model = load_model()
    print_accuracy(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # execute only if run as a script
    main()
