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
from sklearn.utils import shuffle
import glob

KAIST_DATASET_ENABLE = False
AIZU_DATASET_ENABLE = True

n_x = 128
n_y = 64
X_shape = (-1, n_y, n_x, 3)

# AIZU DATASET
example_num_AIZU = 416
total_ex_AIZU = 512
# KAIST DATASET
total_ex_KAIST = 720
# cross validation part on KAIST DATASET
test_part = 2  # from 1->5
start_test = int(total_ex_KAIST / 5 * (test_part - 1))
end_test = int(total_ex_KAIST / 5 * test_part)

max_PSNR = 99.0
max_distance = 1.0
np.set_printoptions(threshold=sys.maxsize)


def get_dataset_AIZU():
    sys.stdout.write('///////////////Loading Dataset AIZU//////////////////\n')
    sys.stdout.flush()
    # list files
    distance_files = glob.glob("data/AIZU-dataset/*.mat")
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


def get_dataset_KAIST(distance_file):
    sys.stdout.write('////////////////Loading Dataset KAIST////////////////////\n')
    sys.stdout.flush()
    # part 1: Load all data:
    # MOS
    MOS = spio.loadmat('data/MOS.mat', squeeze_me=True)
    y_data = MOS['MOS']
    # distance
    distance = spio.loadmat(distance_file, squeeze_me=True)
    distance = distance['distance']
    # PSNR
    PSNR = spio.loadmat('data/PSNR_KAIST.mat', squeeze_me=True)
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
    X_test = X_data[start_test:end_test, :, :, :]
    y_test = y_data[start_test:end_test]
    X_train_1 = X_data[0:start_test, :, :, :]
    X_train_2 = X_data[end_test:, :, :, :]
    X_train = np.concatenate((X_train_1, X_train_2), axis=0)
    y_train_1 = y_data[0:start_test]
    y_train_2 = y_data[end_test:]
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


def combine_dataset():
    # 1: AIZU DATASET
    if AIZU_DATASET_ENABLE:
        X_train_AIZU, y_train_AIZU, X_test_AIZU, y_test_AIZU = get_dataset_AIZU()
    # 2: KAIST DATASET
    if KAIST_DATASET_ENABLE:
        # list file
        distance_files = glob.glob("data/distance/*.mat")
        X_train_KAIST = np.array([])
        y_train_KAIST = np.array([])
        X_test_KAIST = np.array([])
        y_test_KAIST = np.array([])
        for file in distance_files:
            X_train, y_train, X_test, y_test = get_dataset_KAIST(file)
            if X_train_KAIST.size == 0:
                X_train_KAIST, y_train_KAIST, X_test_KAIST, y_test_KAIST = X_train, y_train, X_test, y_test
            else:
                X_train_KAIST = np.concatenate((X_train_KAIST, X_train), axis=0)
                y_train_KAIST = np.concatenate((y_train_KAIST, y_train), axis=0)
                X_test_KAIST = np.concatenate((X_test_KAIST, X_test), axis=0)
                y_test_KAIST = np.concatenate((y_test_KAIST, y_test), axis=0)

        # flip KAIST dataset (horizontally)
        PSNR_temp = X_train_KAIST[:, :, :, 0]
        PSNR_temp = np.flip(PSNR_temp, axis=2)
        X_temp = X_train_KAIST
        X_temp[:, :, :, 0] = PSNR_temp
        X_train_KAIST = np.concatenate((X_train_KAIST, X_temp), axis=0)
        y_flip = y_train_KAIST
        y_train_KAIST = np.concatenate((y_train_KAIST, y_flip), axis=0)
        PSNR_temp = X_test_KAIST[:, :, :, 0]
        PSNR_temp = np.flip(PSNR_temp, axis=2)
        X_temp = X_test_KAIST
        X_temp[:, :, :, 0] = PSNR_temp
        X_test_KAIST = np.concatenate((X_test_KAIST, X_temp), axis=0)
        y_flip = y_test_KAIST
        y_test_KAIST = np.concatenate((y_test_KAIST, y_flip), axis=0)

        # todo: remove comment flip vertically
        # flip KAIST dataset (vertically)
        PSNR_temp = X_train_KAIST[:, :, :, 0]
        PSNR_temp = np.flip(PSNR_temp, axis=1)
        X_temp = X_train_KAIST
        X_temp[:, :, :, 0] = PSNR_temp
        X_train_KAIST = np.concatenate((X_train_KAIST, X_temp), axis=0)
        y_flip = y_train_KAIST
        y_train_KAIST = np.concatenate((y_train_KAIST, y_flip), axis=0)
        PSNR_temp = X_test_KAIST[:, :, :, 0]
        PSNR_temp = np.flip(PSNR_temp, axis=1)
        X_temp = X_test_KAIST
        X_temp[:, :, :, 0] = PSNR_temp
        X_test_KAIST = np.concatenate((X_test_KAIST, X_temp), axis=0)
        y_flip = y_test_KAIST
        y_test_KAIST = np.concatenate((y_test_KAIST, y_flip), axis=0)

    # # ONLY FOR DEBUG PURPOSE (KAIST DATASET)
    # plot KAIST dataset ratio = PSNR/(MOS*10)

    PSNR_temp = X_train_KAIST[:, 32, 64, 0]
    y_temp = y_train_KAIST.reshape((np.size(y_train_KAIST),))
    ratio = np.divide(PSNR_temp * max_PSNR, y_temp * y_temp)
    PSNR_temp = X_test_KAIST[:, 32, 64, 0]
    y_temp = y_test_KAIST.reshape((np.size(y_test_KAIST),))
    ratio = np.divide(PSNR_temp * max_PSNR, y_temp * y_temp)

    print(np.shape(ratio))
    plt.plot(ratio)
    plt.ylabel('RATIO')
    plt.title('RATIO between PSNR/MOS')
    plt.show()

    # Case 1: test case for only AIZU dataset
    if AIZU_DATASET_ENABLE and not KAIST_DATASET_ENABLE:
        X_train = X_train_AIZU
        y_train = y_train_AIZU
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test = X_test_AIZU
        y_test = y_test_AIZU

    # Case 2: test case for only KAIST dataset
    elif not AIZU_DATASET_ENABLE and KAIST_DATASET_ENABLE:
        X_train = X_train_KAIST
        y_train = y_train_KAIST
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test = X_test_KAIST
        y_test = y_test_KAIST

    # Common Case: combine all dataset from KAIST and AIZU
    elif AIZU_DATASET_ENABLE and KAIST_DATASET_ENABLE:
        X_train = np.concatenate((X_train_AIZU, X_train_KAIST), axis=0)
        y_train = np.concatenate((y_train_AIZU, y_train_KAIST), axis=0)
        # shuffle training set before training
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test = np.concatenate((X_test_AIZU, X_test_KAIST), axis=0)
        y_test = np.concatenate((y_test_AIZU, y_test_KAIST), axis=0)

    print("****************FINAL DATASET SHAPE**********************")
    print("X_train shape:" + str(np.shape(X_train)))
    print("Y_train shape:" + str(np.shape(y_train)))
    print("X_test shape:" + str(np.shape(X_test)))
    print("Y_test shape:" + str(np.shape(y_test)))
    return X_train, y_train, X_test, y_test


def main():
    # main
    X_trai, y_trai, X_tes, y_tes = combine_dataset()
    with h5py.File('dataset/X_train.h5', 'w') as hf:
        hf.create_dataset("X_train", data=X_trai)
    with h5py.File('dataset/y_train.h5', 'w') as hf:
        hf.create_dataset("y_train", data=y_trai)
    with h5py.File('dataset/X_test.h5', 'w') as hf:
        hf.create_dataset("X_test", data=X_tes)
    with h5py.File('dataset/y_test.h5', 'w') as hf:
        hf.create_dataset("y_test", data=y_tes)


if __name__ == "__main__":
    # execute only if run as a script
    main()
