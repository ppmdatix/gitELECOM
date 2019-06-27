from loadingCGAN.novgan import Novgan
from load_data.load_data import load_data
from utils.config_novgan import nrows, place
from utils.config_novgan import balanced_train_size, shuffle, cv_size
from loadingCGAN.utils_cgan import creating_dico_index



# DATA
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

novgan = Novgan(dico_index=dico_index)