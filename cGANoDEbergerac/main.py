from loadingCGAN.cgan import Cgan, switching_gans
from loadingCGAN.mlp import Mlp
from evaluation.evaluation import evaluate
import numpy as np
from load_data.load_data import load_data

# Parameters
attack_mode = None
epochs = 200
number_of_gans = 1
number_of_switch = 1

# DATA
x_train, y_train, x_balanced_train, y_balanced_train, x_test, y_test = load_data()
data_dim = x_train.shape[1]
print("\n  \n \n "*2)
print("Train data shape is {}".format(x_balanced_train.shape))
print("\n  \n \n "*2)

########
# CGAN #
########


cgans = [Cgan(data_dim=data_dim, spectral_normalisation=False) for _ in range(number_of_gans)]
for _ in range(number_of_switch):
    for cgan in cgans:
        cgan.train(x_train=x_balanced_train,  # x_train,
                   y_train=y_balanced_train,  # ,y_train
                   epochs=epochs,
                   print_recap=False,
                   reload_images_p=.99,
                   show_past_p=.95)
    if number_of_gans > 1:
        switching_gans(cgans)


cgan = cgans[0]
cgan.plot_learning()

examples = 100

result_cgan = evaluate(y_true=y_test, y_pred=cgan.predict(x=x_test))
generated_one = cgan.generate(number=examples, labels=np.ones(examples))
generated_zero = cgan.generate(number=examples, labels=np.zeros(examples))
#############
# Classical #
#############
mlp = Mlp(data_dim=data_dim)
d_loss_classical = mlp.train(x_train=x_balanced_train,
                             y_train=y_balanced_train,
                             epochs=epochs*number_of_switch)

result_mlp = evaluate(y_true=y_test, y_pred=mlp.predict(x=x_test))
result_mlp_fooling = evaluate(y_true=np.zeros(examples+examples),
                              y_pred=mlp.predict(np.concatenate((generated_one, generated_zero))))


print(result_cgan)

print(result_mlp)

print(result_mlp_fooling)