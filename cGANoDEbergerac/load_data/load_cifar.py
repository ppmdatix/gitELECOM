from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def load_cifar(cv_size=.1):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.reshape(y_train.shape[0])  # somehow y_train comes as a 2D nx1 matrix
    y_test = y_test.reshape(y_test.shape[0])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train, x_train_cv = train_test_split(x_train, test_size=cv_size)

    x_train = x_train.reshape((x_train.shape[0], 32*32*3))
    x_train_cv = x_train_cv.reshape((x_train_cv.shape[0], 32*32*3))
    x_test = x_test.reshape((x_test.shape[0], 32*32*3))
    data_dim = x_train.shape[1]
    return x_train, x_test, y_train, y_test, x_train_cv, data_dim


def draw_img(x_train, i):
    im = x_train[i]
    plt.imshow(im)
    plt.axis('on')
    plt.show()
