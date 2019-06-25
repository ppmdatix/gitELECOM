from loadingCGAN.cgan import Cgan, switching_gans
from loadingCGAN.mlp import Mlp
from evaluation.evaluation import evaluate
from learning import learning
import numpy as np
from load_data.load_data import load_data
from utils.config import epochs, number_of_gans, switches, latent_dim, nrows, dropout, leaky_relu
from utils.config import examples, reload_images_p, show_past_p, smooth_zero, smooth_one, save_model



# DATA
x_train, x_train_cv, y_train, y_train_cv, x_balanced_train, y_balanced_train, x_test, y_test = load_data(place="work",
                                                                                                         nrows=nrows,
                                                                                                         cv_size=.1,
                                                                                                         log_transform=True)
x_balanced_train, y_balanced_train = x_balanced_train[:5000], y_balanced_train[:5000]

data_dim = x_train.shape[1]
print(y_train_cv.sum())


########
# CGAN #
########
cgans = [Cgan(data_dim=data_dim, latent_dim=latent_dim,
              spectral_normalisation=False,
              weight_clipping=False, verbose=True,
              activation="tanh", dropout=dropout, leaky_relu=leaky_relu) for _ in range(number_of_gans)]


cgans = learning(cgans=cgans, x=x_balanced_train, y=y_balanced_train, x_cv=x_train_cv,
                 y_cv=y_train_cv, number_of_gans=number_of_gans,
                 epochs=epochs, switches=switches, print_mode=False, mode_d_loss=True,
                 reload_images_p=reload_images_p, show_past_p=show_past_p,
                 smooth_zero=smooth_zero, smooth_one=smooth_one)


cgan = cgans[0]
if save_model:
    cgan.save_model(location="save_models/models/", model_name="test1")
# cgano = cgans[number_of_gans - 1]
# cgano.load_model(location="save_models/models/", model_name="test1")

cgan.plot_learning()

result_cgan = evaluate(y_true=y_test, y_pred=cgan.predict(x=x_test))
generated_one = cgan.generate(number=examples, labels=np.ones(examples))
generated_zero = cgan.generate(number=examples, labels=np.zeros(examples))
#################
# Classical MLP #
#################
mlp = Mlp(data_dim=data_dim)
d_loss_classical = mlp.train(x_train=x_balanced_train,
                             y_train=y_balanced_train,
                             epochs=epochs*(switches+1))

result_mlp = evaluate(y_true=y_test, y_pred=mlp.predict(x=x_test))

print("CGAN result")
print(result_cgan)
print("\n"*2 + "MLP result")
print(result_mlp)
print("\n"*2 + "MLP fooled by attacker attacking")
print((examples - sum(mlp.predict(generated_one)) ) / examples)
print("\n"*2 + "MLP fooled by attacker not attacking")
print((examples - sum(mlp.predict(generated_zero)) )/ examples)
