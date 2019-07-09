import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def load_cifar(x_train_size=None):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_test = (x_test.astype(np.float32) - 127.5)/127.5
    x_train, x_train_cv = train_test_split(x_train, test_size=.1)
    # x_train = x_train.reshape(int(50000*.9), 32*32*3)
    x_train_cv = x_train_cv.reshape(int(50000*.1), 32*32*3)
    x_test = x_test.reshape(10000, 32*32*3)
    if x_train_size is not None:
        x_train = x_train[:x_train_size]
    data_dim = x_train.shape[1]
    return x_train, x_test, x_train_cv, data_dim




def draw_img(x_train, i):
    plt.figure()
    im = x_train[i]
    # c = y_train[i]
    plt.imshow(im)
    # plt.title("Class %d (%s)" % (c, class_name[c]))
    plt.axis('on')
    plt.tight_layout()
    plt.show()
    plt.close()

def draw_sample(x_train, n, rows=4, cols=4, imfile=None, fontsize=12):
    for i in range(0, rows*cols):
        plt.subplot(rows, cols, i+1)
        im = x_train[n+i].reshape(32,32,3)
        plt.imshow(im, cmap='gnuplot2')
        # plt.title("{}".format(class_name[y_train[n+i]]), fontsize=fontsize)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.6, hspace=0.01)
        #plt.subplots_adjust(hspace=0.45, wspace=0.45)
        #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if imfile:
        plt.savefig(imfile)
    plt.show()

x_train, x_test, x_train_cv, data_dim = load_cifar(x_train_size=None)
x_train = x_train.reshape(x_train.shape[0], 3, 32, 32).transpose([0,2,3,1])
x = x_train[0]
print(x)
plt.imshow(x)
plt.show()