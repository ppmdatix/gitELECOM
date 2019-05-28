import numpy as np
import sys
#sys_path = "/Users/ppx/Desktop/gitELECOM/IDSGAN"
sys_path = "/home/peseux/Desktop/gitELECOM/IDSGAN/"
sys.path.insert(0, sys_path)
from loading.loadingKDD import loadingKDD


def load_data(attack_mode=None, test_size=3000, nrows=10000000, attack=None):

    # DATA
    x_train, y_train, cat_col = loadingKDD(nrows=nrows, attack_mode=attack_mode,
                                           attack=attack, force_cat_col=None)
    x_test, y_test, _ = loadingKDD(nrows=nrows,
                                   attack_mode=attack_mode, attack=attack,
                                   path="/home/peseux/Desktop/gitELECOM/NSL-KDD/KDDTest+.txt",
                                   force_cat_col=cat_col)

    zero_index_train = [i for y, i in zip(y_train, range(len(y_train))) if y == 0]
    one_index_train = [i for y, i in zip(y_train, range(len(y_train))) if y == 1]
    balanced_size = min(len(zero_index_train), len(one_index_train))
    if attack_mode is None:
        x_balanced_train = np.concatenate((x_train[zero_index_train][:balanced_size],
                                           x_train[one_index_train][:balanced_size]))
        y_balanced_train = np.concatenate((y_train[zero_index_train][:balanced_size],
                                           y_train[one_index_train][:balanced_size]))
    elif attack_mode is False:
        x_balanced_train = x_train[zero_index_train]
        y_balanced_train = y_train[zero_index_train]
    elif attack_mode:
        x_balanced_train = x_train[one_index_train]
        y_balanced_train = y_train[one_index_train]
    else:
        raise Exception("attack_mode should be True, False or None")

    return x_train, y_train, x_balanced_train, y_balanced_train, x_test, y_test
