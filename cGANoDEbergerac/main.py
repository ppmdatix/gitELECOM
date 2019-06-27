from loadingCGAN.cgan import Cgan
from loadingCGAN.mlp import Mlp
from evaluation.evaluation import evaluate
from learning import learning
import numpy as np
from load_data.load_data import load_data
from utils.config import epochs, number_of_gans, switches, latent_dim, nrows, dropout, leaky_relu, place, activation, noise
from utils.config import examples, reload_images_p, show_past_p, smooth_zero, smooth_one, save_model, balanced_train_size
from utils.config import spectral_normalisation, shuffle, cv_size



# DATA
x_train, x_train_cv, y_train, y_train_cv, x_balanced_train, y_balanced_train, x_test, y_test = load_data(place=place,
                                                                                                         nrows=nrows,
                                                                                                         cv_size=cv_size,
                                                                                                         log_transform=True,
                                                                                                         shuffle=shuffle)
if balanced_train_size is not None:
    x_balanced_train, y_balanced_train = x_balanced_train[:balanced_train_size], y_balanced_train[:balanced_train_size]

data_dim = x_train.shape[1]


########
# CGAN #
########
cgans = [Cgan(data_dim=data_dim, latent_dim=latent_dim,
              spectral_normalisation=spectral_normalisation,
              weight_clipping=False, verbose=True,noise=noise,
              activation=activation, dropout=dropout, leaky_relu=leaky_relu) for _ in range(number_of_gans)]


cgans = learning(cgans=cgans,
                 x=x_train,# x_balanced_train
                 y=y_train,# y_balanced_train
                 x_cv=x_train_cv,
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
print("\n"*4 + "="*15 + "\n" + "CGAN result")
print(result_cgan)


#################
# Classical MLP #
#################
mlp = Mlp(data_dim=data_dim, verbose=False)
d_loss_classical = mlp.train(x_train=x_train,# x_balanced_train
                             y_train=y_train,# y_balanced_train
                             epochs=epochs*(switches+1))

result_mlp = evaluate(y_true=y_test, y_pred=mlp.predict(x=x_test))

print("\n"*2 + "="*15 + "\n" + "MLP result")
print(result_mlp)



generated_one = cgan.generate(number=examples, labels=np.ones(examples))
generated_zero = cgan.generate(number=examples, labels=np.zeros(examples))
mlp_one = int(sum(mlp.predict(generated_one)))
mlp_zero = int(sum(mlp.predict(generated_zero)))
print("\n"*2 + "MLP fooled by attacker attacking ?")
print("MLP predicts " + str(mlp_one) + " 1label on " + str(examples) + " examples")
print("\n"*2 + "MLP fooled by attacker not attacking ?")
print("MLP predicts " + str(mlp_zero) + " 1label on " + str(examples) + " examples")
