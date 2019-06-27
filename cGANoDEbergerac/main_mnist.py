from loadingCGAN.swagan import Swagan
from loadingCGAN.utils_cgan import plot_images
from learning_mnist import learning_mnist
import numpy as np
from utils.config_mnist import epochs, number_of_gans, switches, latent_dim, x_train_size
from utils.config_mnist import smooth_zero, smooth_one, dropout, leaky_relu, noise, activation
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


# DATA
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


########
# SWAGAN #
########
swagans = [Swagan(data_dim=data_dim, latent_dim=latent_dim, leaky_relu=leaky_relu, dropout=dropout,
                  spectral_normalisation=False,
                  weight_clipping=False, verbose=True,
                  activation=activation,
                  noise=noise) for _ in range(number_of_gans)]


swagans = learning_mnist(swagans=swagans, x=x_train, x_cv=x_train_cv,
                         number_of_gans=number_of_gans,
                         epochs=epochs, switches=switches, print_mode=True,
                         smooth_zero=smooth_zero, smooth_one=smooth_one, title="all_data_no_switch")

swagan = swagans[0]
swagan.plot_learning()
# cgano = cgans[number_of_gans - 1]
# cgano.load_model(location="save_models/models/", model_name="test1")


images = swagan.generate(100)
plot_images(images.reshape(100, 28,28))

