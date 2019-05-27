from loadingCGAN.cgan import Cgan
from loadingCGAN.mlp import Mlp
from evaluation.evaluation import evaluate
import numpy as np
import sys
#sys_path = "/Users/ppx/Desktop/gitELECOM/IDSGAN"
sys_path = "/home/peseux/Desktop/gitELECOM/IDSGAN/"
sys.path.insert(0, sys_path)
from loading.loadingKDD import loadingKDD

# Parameters
attack_mode = None
test_size = 3000

# DATA
x_train, y_train, cat_col = loadingKDD(nrows=10000000, attack_mode=attack_mode, attack=None, force_cat_col=None)
x_test, y_test, _ = loadingKDD(nrows=10000000,
                               attack_mode=attack_mode, attack=None,
                               path="/home/peseux/Desktop/gitELECOM/NSL-KDD/KDDTest+.txt",
                               force_cat_col=cat_col)

zero_index_train = [i for y, i in zip(y_train, range(len(y_train))) if y == 0]
zero_index_test = [i for y, i in zip(y_test, range(len(y_test))) if y == 0]
one_index_train = [i for y, i in zip(y_train, range(len(y_train))) if y == 1]
one_index_test = [i for y, i in zip(y_test, range(len(y_test))) if y == 1]
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
print(x_balanced_train.shape)


data_dim = x_train.shape[1]

########
# CGAN #
########
cgan = Cgan(data_dim=data_dim)
cv_loss, d_loss, g_loss = cgan.train(x_train=x_balanced_train,
                                     y_train=y_balanced_train,
                                     epochs=10,
                                     print_recap=False)


result_cgan = evaluate(y_true=y_test, y_pred=cgan.predict(x=x_test))
generator, discriminator, combined = cgan.return_models()

#############
# Classical #
#############
mlp = Mlp(data_dim=data_dim)
mlp.train(x_train=x_balanced_train,
          y_train=y_balanced_train,
          epochs=10)

result_mlp = evaluate(y_true=y_test, y_pred=mlp.predict(x=x_test))


print(result_cgan)

print(result_mlp)