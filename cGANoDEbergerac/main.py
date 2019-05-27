from loadingCGAN.cgan import Cgan
from loadingCGAN.mlp import Mlp
from evaluation.evaluation import evaluate
import numpy as np
import sys
sys_path = "/Users/ppx/Desktop/gitELECOM/IDSGAN"
# sys_path = "/home/peseux/Desktop/gitELECOM/IDSGAN/"
sys.path.insert(0, sys_path)
from loading.loadingKDD import loadingKDD

# Parameters
attack_mode = None
test_size = 3000

# DATA
X, Y, colnames = loadingKDD(nrows=10000000, attack_mode=attack_mode, attack=None)
x_test, y_test = X[:test_size], Y[:test_size]
x_train, y_train = X[:-test_size], Y[:-test_size]
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
cgan.train(x_train=x_balanced_train,
           y_train=y_balanced_train,
           epochs=1000)

cgan = Cgan(data_dim=data_dim)
cv_loss, d_loss, g_loss = cgan.train(x_train=x_balanced_train,
                                     y_train=y_balanced_train,
                                     epochs=100)


result_cgan = evaluate(y_true=y_test, y_pred=cgan.predict(x_test=x_test))
generator, discriminator, combined = cgan.return_models()

#############
# Classical #
#############
mlp = Mlp(data_dim=data_dim)
mlp.train(x_train=x_balanced_train,
          y_train=y_balanced_train,
          epochs=10000)

result_mlp = evaluate(y_true=y_test, y_pred=mlp.predict(x_test=x_test))


print(result_cgan)

print(result_mlp)