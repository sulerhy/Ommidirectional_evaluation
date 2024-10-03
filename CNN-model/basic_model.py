"""
Copyright (C) Hoang Pham Duc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Hoang Pham Duc <phamduchoangeee@gmail.com>, May 2020
"""

import sys
import json
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D, Conv1D
from keras.layers import Conv2D, MaxPooling2D
import utils
from utils import savetxt
import matplotlib.pyplot as plt

# global static variables
n_x = 128
n_y = 64
X_shape = (-1, n_y, n_x, 3)

big_epoch = 1
epochs_num = 100
batch_size = 32


def generate_optimizer():
    return keras.optimizers.Adam()


def compile_model(model):
    model.compile(loss='mean_squared_error',
                  optimizer=generate_optimizer())


def generate_model():
    sys.stdout.write('Loading new model\n\n')
    sys.stdout.flush()

    model = Sequential()

    # Conv1 64 128 (3) => 62 126 (3)
    model.add(Conv2D(25, (3, 3), input_shape=X_shape[1:]))
    model.add(Activation('relu'))
    # Pool2 62 126 (3) => 31 63 (3)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv2 64 128 (3) => 62 126 (3)
    model.add(Conv2D(20, (3, 3), ))
    model.add(Activation('relu'))
    # Pool2 62 126 (3) => 31 63 (3)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # FC layers 31 63 (3) => 589
    model.add(Flatten())
    model.add(Dropout(0.2))

    # FC layers 5859 => 8
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # FC layers 5859 => 8
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # FC layers 5859 => 8
    model.add(Dense(32, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # FC layers 5859 => 1
    model.add(Dense(1))
    model.add(Activation('relu'))
    # compile has to be done impurely
    compile_model(model)

    with open('./models/convnet_model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
        outfile.close()

    return model


def train(model, X_train, y_train, X_test, y_test):
    sys.stdout.write('Training model\n\n')
    sys.stdout.flush()

    # train each iteration individually to back up current state
    # safety measure against potential crashes

    epoch_count = 0
    while epoch_count < big_epoch:
        epoch_count += 1
        sys.stdout.flush()
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            epochs=epochs_num,
                            verbose=2)
        sys.stdout.write('Epoch {} done, saving model to file\n\n'.format(epoch_count))
        sys.stdout.flush()
        model.save_weights('./models/convnet_weights.h5')

    # print loss and val_loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # debugging(model)
    # savetxt('result/X_train.csv', X_train[3,:,:,0])
    # savetxt('result/predict_test.csv', test_predict)
    sys.stdout.write("Saving Done, training complete!")
    return model


def debugging(model):
    i = 1
    for layer in model.layers:
        print("Layer:" + str(i) + "   ///////////////////////////////////////////////////////////////////")
        print(layer.output)
        i += 1


def main():
    sys.stdout.write('Omnidirectional Image Assessment!\n\n')
    sys.stdout.flush()
    X_train, y_train, X_test, y_test = utils.load_all_data()
    savetxt('result/Y_train.csv', y_train)
    savetxt('result/Y_test.csv', y_test)
    # debugging
    savetxt('result/test.csv', X_train[1, :, :, 0])

    model = generate_model()
    model = train(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # execute only if run as a script
    main()
