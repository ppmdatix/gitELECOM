import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

def load_mnist(x_train_size):
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_test = (x_test.astype(np.float32) - 127.5)/127.5
    x_train, x_train_cv = train_test_split(x_train, test_size=.1)
    x_train = x_train.reshape(int(60000*.9), 784)
    x_train_cv = x_train_cv.reshape(int(60000*.1), 784)
    x_test = x_test.reshape(10000, 784)
    if x_train_size is not None:
        x_train = x_train[:x_train_size]
    data_dim = x_train.shape[1]
    return x_train, x_test, x_train_cv, data_dim