import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten
from tensorflow.keras import Sequential

# def leakyrelu(x):
#     return relu(x, alpha=0.01)

def neural_ce():
    # Creating the Neural network model
    learning_rate = 0.001
    model = Sequential()
    # Layers :

    # 1. First Convolutional Layer
    model.add(Conv1D(64, kernel_size=10, strides=1, input_shape=(784, 1), activation='relu'))
    # dropout layer
    model.add(Dropout(0.25))
    # pooling layer
    model.add(MaxPooling1D(pool_size=2))

    # 2. Second Convolutional layer
    model.add(Conv1D(64, kernel_size=7, strides=1, activation='relu'))
    # dropout layer
    model.add(Dropout(0.25))
    # pooling layer
    model.add(MaxPooling1D(pool_size=2))

    # 3. Third Convolutional layer
    model.add(Conv1D(128, kernel_size=5, strides=1, activation='relu'))
    # dropout layer
    model.add(Dropout(0.25))
    # pooling layer
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    # 4. Fully Connected layer
    model.add(Dense(64, activation='relu'))
    # 5. Output
    model.add(Dense(10, activation="softmax"))
    model.summary()
    opt = SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def neural_mse():
    # Creating the Neural network model
    learning_rate = 0.001
    model = Sequential()
    # Layers :

    # 1. First Convolutional Layer
    model.add(Conv1D(64, kernel_size=10, strides=1, input_shape=(784, 1), activation='relu'))
    # dropout layer
    model.add(Dropout(0.25))
    # pooling layer
    model.add(MaxPooling1D(pool_size=2))

    # 2. Second Convolutional layer
    model.add(Conv1D(64, kernel_size=7, strides=1, activation='relu'))
    # dropout layer
    model.add(Dropout(0.25))
    # pooling layer
    model.add(MaxPooling1D(pool_size=2))

    # 3. Third Convolutional layer
    model.add(Conv1D(128, kernel_size=5, strides=1, activation='relu'))
    # dropout layer
    model.add(Dropout(0.25))
    # pooling layer
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    # 4. Fully Connected layer
    model.add(Dense(64, activation='relu'))
    # 5. Output
    model.add(Dense(10, activation="softmax"))
    model.summary()
    opt = SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    return model


def preprocessAndRun():
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
    # Pixel values between 0 and 255, so we will use MinMaxScaler to scale it to [0,1].
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    y_train = norm.fit_transform(y_train)

    # for later use
    X_test = norm.fit_transform(X_test)
    y_test = norm.fit_transform(y_test)

    # Split for kfold validation
    mses = list()
    cross_ent = list()
    acc1 = list()
    acc2 = list()
    kf = KFold(n_splits=5, random_state=None, shuffle=False)  # Maybe shuffle. Check later

    for k, (train_ind, test_ind) in enumerate(kf.split(X_train, y_train)):

        X_train_new, X_test_new = X_train[train_ind, :], X_train[test_ind, :]
        y_train_new, y_test_new = y_train[train_ind], y_train[test_ind]

        # we need 3d vectors
        X_train_new = np.reshape(X_train_new,(X_train_new.shape[0], X_train_new.shape[1], 1))
        X_test_new = np.reshape(X_test_new,(X_test_new.shape[0], X_test_new.shape[1], 1))

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Early stopping for A2 final subquestion.
        callback = EarlyStopping(monitor="val_accuracy", mode="max", min_delta=0, patience=5, verbose=1)

        ce_model = neural_ce()
        history1 = ce_model.fit(X_train_new, y_train_new, epochs=200, batch_size=32,
                                validation_data=(X_test_new, y_test_new), verbose=1)

        loss1, accuracy1 = ce_model.evaluate(X_test, y_test, verbose=0)

        mse_model = neural_mse()
        history2 = mse_model.fit(X_train_new, y_train_new, epochs=200, batch_size=32,
                                 validation_data=(X_test_new, y_test_new), callbacks=[callback], verbose=1)

        loss2, accuracy2 = mse_model.evaluate(X_test, y_test, verbose=0)

        # stores scores
        # cross entropy accuracy
        acc1.append(accuracy1)
        # mse accuracy
        acc2.append(accuracy2)

        # for cross entropy loss
        cross_ent.append(loss1)
        print(f"\nNumber {k} Fold's Cross entropy score is {loss1}")

        # for mse loss
        mses.append(loss2)
        print(f"\nNumber {k} Fold's MSE score is {loss2}")

    # print("Accuracy for Cross Entropy loss is : ", np.mean(acc1) * 100)
    print("Accuracy for MSE loss is : ", np.mean(acc2) * 100)

    # for cross entropy loss
    print("Cross-Entropy is : ", np.mean(cross_ent))

    # for mse loss
    print("MSE is : ", np.mean(mses))

    # Plotting!
    plt.figure(0)
    plt.subplot(2, 2, 1)
    plt.plot(history1.history['accuracy'], label='Accuracy (train)')
    plt.plot(history1.history['val_accuracy'], label='Accuracy (test)')
    plt.title("Accuracy with Cross Entropy loss")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history2.history['accuracy'], label='Accuracy (train)')
    plt.plot(history2.history['val_accuracy'], label='Accuracy (test)')
    plt.title("Accuracy with MSE loss")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # plot the cross entropy loss
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(history1.history['loss'], label='Cross entropy (train)')
    plt.plot(history1.history['val_loss'], label='Cross entropy (test)')
    plt.title('Cross Entropy Evaluated')
    plt.xlabel('Epochs')
    plt.ylabel('Error value')
    plt.legend()

    # plot the mse loss
    plt.subplot(2, 2, 2)
    plt.plot(history2.history['loss'], label='MSE (train)')
    plt.plot(history2.history['val_loss'], label='MSE (test)')

    plt.title('MSE Evaluated')
    plt.xlabel('Epochs')
    plt.ylabel('Error value')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    preprocessAndRun()
