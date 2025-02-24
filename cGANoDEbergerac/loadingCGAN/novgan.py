from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from keras.initializers import glorot_uniform
import sys
from matplotlib import pyplot as plt
sys.path.append("..")
from losses.losses_novgan import custom_loss, loss_function_discriminator, hurting_raw
import sys
place = "home"
if place == "work":
    sys_path = "/home/peseux/Desktop/gitELECOM/cGANoDEbergerac/loadingCGAN"
elif place == "home":
    sys_path = "/Users/ppx/Desktop/gitELECOM/cGANoDEbergerac/loadingCGAN"
sys.path.insert(0, sys_path)
from utils_cgan import smoothing_y


def zero_or_one(x):
    if x < .5:
        return 0
    else:
        return 1


class Novgan(object):
    """
    Novgan GAN where loss can be choosed in order to fit the report description

    Working on NSLKDD data
    """
    def __init__(self, data_dim=28, activation="tanh", verbose=True,
                 latent_dim=32,
                 leaky_relu=.1, offset=0, alpha=1., dropout=.2,
                 dico_index=None,
                 noise="normal",
                 smooth_one=.9, smooth_zero=.1, batch_size=128):
        # Input shape
        self.data_dim = data_dim
        self.activation = activation
        self.verbose = verbose
        self.latent_dim = latent_dim
        self.leaky_rely = leaky_relu
        self.offset = offset
        self.alpha = alpha
        self.dropout = dropout
        self.dico_index = dico_index
        self.noise = noise
        self.custom_loss = custom_loss
        self.smooth_one = smooth_one
        self.smooth_zero = smooth_zero
        self.batch_size = batch_size

        self.optimizer = Adam(0.0002, 0.5)
        if self.verbose:
            print("CHOSEN OPTIMIZER IS ADAM")

        # Build and compile the discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = None
        self.gan = self.build_combined()

        self.history = {"d_loss": [], "g_loss": [], "hurting": []}

    def build_generator(self):
        generator = Sequential()
        generator.add(Dense(64, input_dim=self.latent_dim, kernel_initializer=glorot_uniform()))
        generator.add(LeakyReLU(self.leaky_rely))
        generator.add(Dense(128))
        generator.add(LeakyReLU(self.leaky_rely))
        generator.add(Dense(self.data_dim, activation=self.activation))
        # generator.compile(loss="binary_crossentropy", optimizer=self.optimizer)

        if self.verbose:
            print("\n \n Generator Architecture ")
            generator.summary()
        return generator

    def build_discriminator(self):
        """
        The most important part is the custom loss definition
        We redefine the Discriminator loss in order to fit our Novgan proposition

        More info on losses_novgan.py file
        """
        discriminator = Sequential()
        discriminator.add(Dense(18, input_dim=self.data_dim))
        discriminator.add(LeakyReLU(self.leaky_rely))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(12))
        discriminator.add(LeakyReLU(self.leaky_rely))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(10))
        discriminator.add(LeakyReLU(self.leaky_rely))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss=loss_function_discriminator, optimizer=self.optimizer)
        return discriminator

    def build_combined(self):
        """
        The most important part is the custom loss definition
        We redefine the Generator loss in order to fit our Novgan proposition

        More info on losses_novgan.py file
        """

        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan_loss = custom_loss(intermediate_output=x, alpha=self.alpha, offset=self.offset, dico=self.dico_index)
        gan.compile(loss=gan_loss,
                    optimizer=self.optimizer)

        return gan

    def generate(self, number):
        if self.noise == "normal":
            noise = np.random.normal(0, 1, (number, self.latent_dim))
        elif self.noise == "logistic":
            noise = np.random.logistic(0, 1, (number, self.latent_dim))
        generated_traffic = self.generator.predict(noise)
        return generated_traffic

    def train(self, x_train, epochs, print_recap=True):

        d_loss, g_loss, h_evolution = list(), list(), list()
        ones = np.ones((self.batch_size,1))
        zeros = np.zeros((self.batch_size, 1))
        batch_count = int(x_train.shape[0] / self.batch_size)

        for epoch in range(epochs):
            if print_recap:
                sys.stdout.flush()
                print("# Epoch " + str(epoch) + "/"+str(epochs), flush=True)
            d_l, g_l, h = 0, 0, 0
            for _ in (range(batch_count)):
                idx = np.random.randint(0, x_train.shape[0], self.batch_size)
                real_traffic = x_train[idx]
                generated_traffic = self.generate(number=self.batch_size)
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(real_traffic, smoothing_y(ones,
                                                                                          smooth_one=self.smooth_one,
                                                                                          smooth_zero=self.smooth_zero))
                d_loss_fake = self.discriminator.train_on_batch(generated_traffic, zeros)
                d_l += 0.5 * np.add(d_loss_real, d_loss_fake)
                self.discriminator.trainable = False
                g_l += self.gan.train_on_batch(noise, ones)
                h += np.mean(self.hurting(self.generate(number=self.batch_size)))

            d_loss.append(d_l/batch_count)
            g_loss.append(g_l/batch_count)
            h_evolution.append(h/batch_count)

        self.history["d_loss"] = self.history["d_loss"] + d_loss
        self.history["g_loss"] = self.history["g_loss"] + g_loss
        self.history["hurting"] = self.history["hurting"] + h_evolution
        if print_recap:
            self.plot_learning()
        return d_loss, g_loss

    def hurting(self, x, print_mode=False, title="title"):
        output = [hurting_raw(xi, dico=self.dico_index, mode="numpy") for xi in x]
        if print_mode:
            plt.hist(output, label="malveillance")
            plt.title(title)
            plt.show()
            plt.close()
        return np.array(output)

    def plot_learning(self, save_mode=False, title=None):
        plt.plot(self.history["d_loss"], label="discriminator loss")
        plt.plot(self.history["g_loss"], label="generator loss")
        plt.plot(self.history["hurting"], label="hurting generation")
        plt.xlabel("epochs")
        plt.title("NOVGAN on KDD99 : Learning evolution")
        plt.legend()
        if save_mode:
            plt.savefig("tmp/" + title + ".png")
        plt.show()
        plt.close()
        return True

