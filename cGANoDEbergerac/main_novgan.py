from loadingCGAN.novgan import Novgan
from load_data.load_data import load_data
from utils.config_novgan import nrows, place, activation, latent_dim, leaky_relu, alpha, offset, dropout
from utils.config_novgan import epochs, balanced_train_size, shuffle, cv_size, smooth_one, smooth_zero, batch_size
from loadingCGAN.utils_cgan import creating_dico_index
from time import time
from loadingCGAN.utils_cgan import save_time


########
# DATA #
########
x_train, x_train_cv, y_train, y_train_cv, x_balanced_train, y_balanced_train, x_test, y_test, colnames = load_data(place=place,
                                                                                                                   nrows=nrows,
                                                                                                                   cv_size=cv_size,
                                                                                                                   log_transform=True,
                                                                                                                   shuffle=shuffle, return_colnames=True)
if balanced_train_size is not None:
    x_balanced_train, y_balanced_train = x_balanced_train[:balanced_train_size], y_balanced_train[:balanced_train_size]

data_dim = x_train.shape[1]
dico_index = creating_dico_index(colnames=colnames)

##########
# NovGan #
##########
novgan = Novgan(data_dim=data_dim, activation=activation, verbose=True,
                latent_dim=latent_dim,
                leaky_relu=leaky_relu, offset=offset, alpha=alpha, dropout=dropout,
                dico_index=dico_index,
                noise="normal",
                smooth_one=smooth_one, smooth_zero=smooth_zero, batch_size=batch_size)
############
# Training #
############
start = time()
learned = novgan.train(x_train=x_train, epochs=epochs, print_recap=True)
end = time()
duration = end - start
save_time(duration=duration, location="tmp/", title=title)

novgan.plot_learning(save_mode=True, title="novgan")
############
# Plotting #
############
# novgan.hurting(x=x_test, print_mode=True,title="Real Data")
# novgan.hurting(x=novgan.generate(number=len(x_test)),
#                print_mode=True,
#                title="Generated Data")
