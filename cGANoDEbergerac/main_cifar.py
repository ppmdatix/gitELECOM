from loadingCGAN.swagan import Swagan
from learning_mnist import learning_mnist
from utils.config_mnist import epochs, number_of_gans, switches, latent_dim, x_train_size, title
from utils.config_mnist import smooth_zero, smooth_one, dropout, leaky_relu, noise, activation, save_model
from load_data.load_cifar import load_cifar, draw_img
from time import time
from loadingCGAN.utils_cgan import save_time
import numpy as np

########
# DATA #
########
x_train, x_test, y_train, y_test, x_train_cv, data_dim = load_cifar(cv_size=.1)



##########
# SWAGAN #
##########
swagans = [Swagan(data_dim=data_dim, latent_dim=latent_dim, leaky_relu=leaky_relu, dropout=dropout,
                  spectral_normalisation=False,
                  weight_clipping=False, verbose=True,
                  activation=activation,
                  noise=noise) for _ in range(number_of_gans)]

############
# LEARNING #
############
start = time()
swagans = learning_mnist(swagans=swagans, x=x_train, x_cv=x_train_cv,
                         number_of_gans=number_of_gans,
                         epochs=epochs, switches=switches, print_mode=True,
                         smooth_zero=smooth_zero, smooth_one=smooth_one, title=title)
end = time()
duration = end - start
save_time(duration=duration, location="tmp/", title=title)
##################
# SAVING RESULTS #
##################
swagan = swagans[0]
swagan.plot_learning(save_mode=True, title=title)
if save_model:
    swagan.save_model(model_name="title")
