from loadingCGAN.cgan import Cgan
from loadingCGAN.mlp import Mlp
from evaluation.evaluation import evaluate
import numpy as np
import sys
from load_data.load_data import load_data

# Parameters
attack_mode = None
epochs = 100

# DATA
x_train, y_train, x_balanced_train, y_balanced_train, x_test, y_test = load_data()
data_dim = x_train.shape[1]
print("\n  \n \n "*2)
print("Train data shape is {}".format(x_balanced_train.shape))
print("\n  \n \n "*2)



########
# CGAN #
########
cgan = Cgan(data_dim=data_dim)
cv_loss, d_loss, g_loss = cgan.train(x_train=x_balanced_train,
                                     y_train=y_balanced_train,
                                     epochs=epochs,
                                     print_recap=False,
                                     reload_images_p=.95,
                                     show_past_p=.98)


result_cgan = evaluate(y_true=y_test, y_pred=cgan.predict(x=x_test))
generated_one = cgan.generate(number=100, labels=np.ones(100))
generated_zero = cgan.generate(number=100, labels=np.zeros(100))
#############
# Classical #
#############
mlp = Mlp(data_dim=data_dim)
d_loss_classical = mlp.train(x_train=x_balanced_train, y_train=y_balanced_train, epochs=epochs)

result_mlp = evaluate(y_true=y_test, y_pred=mlp.predict(x=x_test))
result_mlp_fooling = evaluate(y_true=np.zeros(100+100), y_pred=mlp.predict(np.concatenate((generated_one, generated_zero))))


print(result_cgan)

print(result_mlp)

print(result_mlp_fooling)