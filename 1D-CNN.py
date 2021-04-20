import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Dropout, MaxPooling1D, AveragePooling1D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu
from argparse import ArgumentParser

def leakyrelu(x):
    return relu(x, alpha=0.01)


def preprocess():
    train = pd.read_csv('mnist_train.csv')
    test = pd.read_csv('mnist_test.csv')

    X_train = train.drop("label", axis=1, inplace=False)
    X_test = test.drop("label", axis=1, inplace=False)

    # Unsigned int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train = train.label
    y_test = test.label

    # Make categorical
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    # 2. Normalization
    # Pixel values between 0 and 255, so we will divide by 255.
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    y_train = norm.fit_transform(y_train)

    # for later use
    X_test = norm.fit_transform(X_test)
    y_test = norm.fit_transform(y_test)

    # Split for kfold validation
    mses = list()
    cross_ent = list()
    acc = list()
    kf = KFold(n_splits=5, random_state=None, shuffle=False)  # Maybe shuffle. Check later

    for k, (train_ind, test_ind) in enumerate(kf.split(X_train, y_train)):

        X_train_new, X_test_new = X_train[train_ind, :], X_train[test_ind, :]
        y_train_new, y_test_new = y_train[train_ind], y_train[test_ind]

        # we need 3d vectors
        X_train_new = np.reshape(X_train_new,(X_train_new.shape[0], X_train_new.shape[1], 1))
        X_test_new = np.reshape(X_test_new,(X_test_new.shape[0], X_test_new.shape[1], 1))

        # Creating the Neural network model
        learning_rate = 0.001
        # Sequential model start
        model = Sequential()
        # Layers :

        # 1. First Convolutional Layer
        model.add(Conv1D(64, kernel_size=10, strides=1, input_shape=(784, 1), activation=leakyrelu))
        # dropout layer
        model.add(Dropout(0.25))
        # pooling layer
        model.add(MaxPooling1D(pool_size=2))

        # 2. Second Convolutional layer
        # model.add(Conv1D(64, kernel_size=3, strides=1, activation=leakyrelu))
        # # dropout layer
        # model.add(Dropout(0.25))
        # # pooling layer
        # model.add(MaxPooling1D(pool_size=2))

        # 3. Third Convolutional layer
        # model.add(Conv1D(128, kernel_size=1, strides=1, activation=leakyrelu))
        # # dropout layer
        # model.add(Dropout(0.25))
        # # pooling layer
        # model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        # 3. Output
        # model.add(Dense(128, activation=leakyrelu))
        # # dropout layer
        # model.add(Dropout(0.5))
        model.add(Dense(64, activation=leakyrelu))
        model.add(Dense(10, activation="softmax"))
        model.summary()
        opt = RMSprop(learning_rate=learning_rate)
        callback = EarlyStopping(monitor="categorical_crossentropy", mode="min", min_delta=0.1, patience=5, verbose=1)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', 'mean_squared_error',
                                                                               'categorical_crossentropy'])

        history = model.fit(X_train_new, y_train_new, epochs=10, batch_size=256,
                            validation_data=(X_test_new, y_test_new), callbacks=[callback], verbose=1)

        loss, accuracy, mse, cross = model.evaluate(X_test_new, y_test_new, verbose=0)
        # stores scores
        acc.append(accuracy)
        mses.append(mse)
        cross_ent.append(cross)
        print(f"Number {k} Fold's MSE score is {mse} and Cross entropy score is {cross}")

    print("MSE is : ", np.mean(mses))
    print("Cross-Entropy is : ", np.mean(cross_ent))

    # Plotting!s
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='Accuracy (train)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (test)')
    plt.title("Accuracy of the model")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # plot the losses
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(history.history['categorical_crossentropy'], label='Cross entropy (train)')
    plt.plot(history.history['val_categorical_crossentropy'], label='Cross entropy (test)')
    plt.title('Cross Entropy Evaluated')
    plt.xlabel('Epochs')
    plt.ylabel('Error value')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history['mean_squared_error'], label='MSE (train)')
    plt.plot(history.history['val_mean_squared_error'], label='MSE (test)')

    plt.title('MSE Evaluated')
    plt.xlabel('Epochs')
    plt.ylabel('Error value')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Parsing the command line arguments
    parser = ArgumentParser(description="CNN")
    preprocess()
