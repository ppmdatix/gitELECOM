from loadingCGAN.swagan_trafic_input import Swagan_trafic_input
from learning_swagan_trafic_input import learning_swagan_trafic_input
from utils.config_swagan_trafic_input import *
import numpy as np
from load_data.load_data import load_data
from time import time
from loadingCGAN.utils_cgan import save_time
from loadingCGAN.utils_cgan import creating_dico_index
from evaluation.evaluation import evaluate
from loadingCGAN.mlp import Mlp
from sklearn.model_selection import train_test_split

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



x_train, x_train_cv = train_test_split(x_train, test_size=cv_size)
x_train_bad, x_train_bad_cv = train_test_split(x_train_bad, test_size=cv_size)


if balanced_train_size is not None:
    x_balanced_train = x_balanced_train[:balanced_train_size]

data_dim = x_train.shape[1]
dico_index = creating_dico_index(colnames=colnames)


##########
# SWAGAN #
##########
swagans = [Swagan_trafic_input(data_dim=data_dim, noise_dim=noise_dim, leaky_relu=leaky_relu, dropout=dropout,
                  spectral_normalisation=False,
                  weight_clipping=False, verbose=True,
                  activation=activation,
                  noise=noise) for _ in range(number_of_gans)]

############
# LEARNING #
############
start = time()
swagans, swagan_base = learning_swagan_trafic_input(swagans=swagans, x=x_train,
                                                    x_cv=x_train_cv, x_bad=x_train_bad, x_bad_cv=x_train_bad_cv,
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
    swagan.save_model(model_name=title)



##############
# EVALUATION #
##############
evaluation = True
if evaluation:
    x_eval = np.concatenate((x_test, x_test_bad))

    result_swagan = evaluate(y_true=np.array([0. for _ in x_test] + [1. for _ in x_test_bad]),
                             y_pred=swagan.predict(x=x_eval))
    print("\n"*4 + "="*15 + "\n" + "SWAGAN result")
    print(result_swagan)
    #################
    # Classical MLP #
    #################
    mlp = Mlp(data_dim=data_dim, verbose=False)
    d_loss_classical = mlp.train(x_train=np.concatenate((x_train, x_train_bad)),# x_balanced_train
                                 y_train=np.array([0. for _ in x_train] + [1. for _ in x_train_bad]),# y_balanced_train
                                 epochs=epochs*(switches+1))
    result_mlp = evaluate(y_true=np.array([0. for _ in x_test] + [1. for _ in x_test_bad]),
                          y_pred=mlp.predict(x=x_eval))

    print("\n"*2 + "="*15 + "\n" + "MLP result")
    print(result_mlp)