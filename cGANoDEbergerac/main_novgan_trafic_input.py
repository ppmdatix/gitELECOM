from loadingCGAN.novgan_trafic_input import Novgan_trafic_input
from load_data.load_data import load_data
from utils.config_novgan_trafic_input import *
from loadingCGAN.utils_cgan import creating_dico_index
from time import time
import numpy as np
from loadingCGAN.utils_cgan import save_time
########
# DATA #
########
x_train, _, y_train, _, x_balanced_train, _, x_test, y_test, colnames = load_data(place=place,
                                                                                  attack_mode=attack_mode,
                                                                                  attack=attack,
                                                                                  nrows=nrows,
                                                                                  cv_size=cv_size,
                                                                                  log_transform=True,
                                                                                  shuffle=shuffle,
                                                                                  return_colnames=True)
x_train_bad = np.array([x for x, y in zip(x_train, y_train) if int(y) == 1])
x_train = np.array([x for x, y in zip(x_train, y_train) if int(y) == 0])

x_test_bad = np.array([x for x, y in zip(x_test, y_test) if int(y) == 1])
x_test = np.array([x for x, y in zip(x_test, y_test) if int(y) == 0])


if balanced_train_size is not None:
    x_balanced_train = x_balanced_train[:balanced_train_size]

data_dim = x_train.shape[1]
dico_index = creating_dico_index(colnames=colnames)

##########
# NovGan #
##########
novgan_trafic_input = Novgan_trafic_input(data_dim=data_dim, activation=activation, verbose=True,
                                          noise_dim=noise_dim,
                                          leaky_relu=leaky_relu, offset=offset, alpha=alpha, dropout=dropout,
                                          dico_index=dico_index,
                                          noise="normal",
                                          smooth_one=smooth_one,
                                          smooth_zero=smooth_zero, batch_size=batch_size)
############
# Training #
############
start = time()
learned = novgan_trafic_input.train(x_train=x_train, epochs=epochs,
                                    x_train_bad=x_train_bad,
                                    print_recap=False)
end = time()
duration = end - start
save_time(duration=duration, location="tmp/", title=title)



############
# Plotting #
############
# novgan_trafic_input.hurting(x=x_test, title="benin test", print_mode=True)
# novgan_trafic_input.hurting(x=x_test_bad, title="malveillant test", print_mode=True)
# novgan_trafic_input.hurting(x=novgan_trafic_input.generate(number=100, x_bad=x_test_bad[:100]),
#                            title="malveillant test", print_mode=True)