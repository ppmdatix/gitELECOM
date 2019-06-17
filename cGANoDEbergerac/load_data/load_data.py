import numpy as np
import sys
from sklearn.model_selection import train_test_split
sys_path = "/Users/ppx/Desktop/gitELECOM/NSL-KDD"
# sys_path = "/home/peseux/Desktop/gitELECOM/NSL-KDD/"
sys.path.insert(0, sys_path)
from loading.loadingKDD import loadingKDD


def load_data(attack_mode=None, nrows=10000000, attack=None, verbose=True, shuffle=False, cv_size=0., place="home"):

    # DATA
    x_train, y_train, cat_col, _ = loadingKDD(nrows=nrows, attack_mode=attack_mode,
                                           attack=attack, force_cat_col=None, place=place)
    x_test, y_test, _, _ = loadingKDD(nrows=nrows,
                                   attack_mode=attack_mode, attack=attack,
                                   path="/home/peseux/Desktop/gitELECOM/NSL-KDD/KDDTest+.txt",
                                   force_cat_col=cat_col, place=place)
    if shuffle:
        idx = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[idx], y_train[idx]

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
    if verbose:
        print("Train data shape is " + str(x_train.shape))
        print("Test data shape is " + str(x_test.shape))
    if cv_size == 0.:
        return x_train, y_train, x_balanced_train, y_balanced_train, x_test, y_test
    else:
        x_train, x_train_cv, y_train, y_train_cv = train_test_split(x_train, y_train, test_size=cv_size)

        return x_train, x_train_cv, y_train, y_train_cv, x_balanced_train, y_balanced_train, x_test, y_test
