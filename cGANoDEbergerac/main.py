from loadingCGAN.cgan import Cgan, switching_gans
from loadingCGAN.mlp import Mlp
from evaluation.evaluation import evaluate
from evaluation.eliminate import eliminate
from learning import learning
import numpy as np
from load_data.load_data import load_data

# Parameters
attack_mode = None
epochs = 32
number_of_gans = 6
number_of_switch = 6

# DATA
x_train, x_train_cv, y_train, y_train_cv, x_balanced_train, y_balanced_train, x_test, y_test = load_data(cv_size=.05)
data_dim = x_train.shape[1]
print("\n  \n \n "*2)
print("Train data shape is {}".format(x_balanced_train.shape))
print("\n  \n \n "*2)

########
# CGAN #
########


cgans = [Cgan(data_dim=data_dim,
              spectral_normalisation=False,
              weight_clipping=False, verbose=False) for _ in range(number_of_gans)]

generators, discriminators = list(), list()
for i in range(number_of_gans):
    generators.append(cgans[i].generator)
    discriminators.append(cgans[i].discriminator)

cgans = learning(cgans=cgans, x=x_balanced_train, y=y_balanced_train, x_cv=x_train_cv,
                 y_cv=y_train_cv, number_of_gans=number_of_gans,
                 epochs=epochs, switches=2)


cgan = cgans[0]
cgan.save_model(location="save_models/models/", model_name="test1")
# cgano = cgans[number_of_gans - 1]
# cgano.load_model(location="save_models/models/", model_name="test1")

cgan.plot_learning()

examples = 100

result_cgan = evaluate(y_true=y_test, y_pred=cgan.predict(x=x_test))
generated_one = cgan.generate(number=examples, labels=np.ones(examples))
generated_zero = cgan.generate(number=examples, labels=np.zeros(examples))
#################
# Classical MLP #
#################
mlp = Mlp(data_dim=data_dim)
d_loss_classical = mlp.train(x_train=x_balanced_train,
                             y_train=y_balanced_train,
                             epochs=epochs*number_of_switch)

result_mlp = evaluate(y_true=y_test, y_pred=mlp.predict(x=x_test))
result_mlp_fooling = evaluate(y_true=np.zeros(examples+examples),
                              y_pred=mlp.predict(np.concatenate((generated_one, generated_zero))))


print("CGAN result")
print(result_cgan)
print("\n"*2 + "MLP result")
print(result_mlp)
print("\n"*2 + "MLP fooled by attacker")
print(result_mlp_fooling)

