from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import initializers
from keras.models import model_from_json
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import sys
place = "home"
if place == "work":
    sys_path = "/home/peseux/Desktop/gitELECOM/spectralNormalisation/"
elif place == "home":
    sys_path = "/Users/ppx/Desktop/gitELECOM/spectralNormalisation"

sys.path.insert(0, sys_path)
if place == "work":
    sys_path = "/home/peseux/Desktop/gitELECOM/cGANoDEbergerac/loadingCGAN"
elif place == "home":
    sys_path = "/Users/ppx/Desktop/gitELECOM/cGANoDEbergerac/loadingCGAN"
sys.path.insert(0, sys_path)
from utils_cgan import smoothing_y


def switching_swagans(list_of_gans):
    print("Let's switch the GANs")
    length = len(list_of_gans)
    sigma = np.random.permutation(length)
    fixed_points = sum([i == sigma[i] for i in range(length)])
    print("There are "+str(fixed_points)+" fixed points")
    generators, discriminators = list(), list()
    for i in range(length):
        generators.append(list_of_gans[i].generator)
        discriminators.append(list_of_gans[i].discriminator)
    for i in range(length):
        list_of_gans[i].discriminator = discriminators[sigma[i]]
        list_of_gans[i].build_combined()
    print("GANs switched")
    return list_of_gans, sigma


class Swagan(object):
    def __init__(self,
                 data_dim=28,
                 latent_dim=32,
                 batch_size=128,
                 leaky_relu=.02,
                 dropout=.4,
                 spectral_normalisation=False,
                 weight_clipping=False,
                 weight_clip=1,
                 verbose=False,
                 activation="tanh",
                 gan_loss="binary_crossentropy",
                 discriminator_loss="binary_crossentropy",
                 noise="normal"):
        # Input shape
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.verbose = verbose
        self.activation = activation
        self.gan_los = gan_loss
        self.discriminator_loss = discriminator_loss
        self.optimizer = Adam(0.0002, 0.5)
        if self.verbose:
            print("CHOSEN OPTIMIZER IS ADAM")
        self.leaky_relu = leaky_relu
        self.dropout = dropout
        self.spectral_normalisation = spectral_normalisation
        self.weight_clipping = weight_clipping
        self.weight_clip = weight_clip
        self.noise = noise
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss=self.discriminator_loss,
                                   optimizer=self.optimizer)

        # Build the generator
        self.generator = self.build_generator()
        self.discriminator.trainable = False
        self.combined = None
        self.build_combined()
        self.history = {"cv_loss": [], "d_loss": [], "g_loss": []}

    def build_generator(self):
        generator = Sequential()
        generator.add(Dense(256, input_dim=self.latent_dim,
                            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(self.leaky_relu))
        generator.add(Dense(512))
        generator.add(LeakyReLU(self.leaky_relu))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(self.leaky_relu))
        generator.add(Dense(784,
                            activation=self.activation))
        generator.compile(loss=self.discriminator_loss, optimizer=self.optimizer)

        if self.verbose:
            print("\n \n Generator Architecture ")
            generator.summary()

        return generator

    def build_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=784,
                                kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(self.leaky_relu))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(self.leaky_relu))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(self.leaky_relu))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(1,
                                activation="sigmoid"))
        discriminator.compile(loss=self.discriminator_loss, optimizer=self.optimizer)
        if self.verbose:
            print("\n \n Discriminator Architecture ")
            discriminator.summary()
        return discriminator

    def build_combined(self):
        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        self.combined = Model(inputs=gan_input, outputs=gan_output)
        self.combined.compile(loss=self.discriminator_loss, optimizer=self.optimizer)

    def generate(self, number):
        if self.noise == "normal":
            noise = np.random.normal(0, 1, (number, self.latent_dim))
        elif self.noise == "logistic":
            noise = np.random.logistic(0, 1, (number, self.latent_dim))
        generated_traffic = self.generator.predict(noise)
        return generated_traffic

    def train(self, x_train, epochs, print_recap=True, smooth_zero=.1, smooth_one=.9):
        """

        :param x_train:
        :param epochs:
        :param print_recap:
        :param smooth_zero:
        :param smooth_one:
        :return:
        """
        d_loss, g_loss = list(), list()
        ones = np.ones((self.batch_size,1))
        zeros = np.zeros((self.batch_size, 1))
        batch_count = int(x_train.shape[0] / self.batch_size)

        for _ in tqdm(range(epochs)):
            #  Train Discriminator
            # Select a random half batch of images
            d_l, g_l = 0, 0
            for _ in (range(batch_count)):
                idx = np.random.randint(0, x_train.shape[0], self.batch_size)
                real_traffic = x_train[idx]
                generated_traffic = self.generate(number=self.batch_size)
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(real_traffic, smoothing_y(ones,
                                                                               smooth_one=smooth_one,
                                                                               smooth_zero=smooth_zero))
                d_loss_fake = self.discriminator.train_on_batch(generated_traffic, zeros)
                d_l += 0.5 * np.add(d_loss_real, d_loss_fake)
                self.discriminator.trainable = False
                g_l += self.combined.train_on_batch(noise, ones)

            d_loss.append(d_l/batch_count)
            g_loss.append(g_l/batch_count)

        self.history["d_loss"] = self.history["d_loss"] + d_loss
        self.history["g_loss"] = self.history["g_loss"] + g_loss
        if print_recap:
            self.plot_learning()
        return d_loss, g_loss

    def evaluate(self, x, batch_size=None):
        """

        :param x:
        :param batch_size:
        :return:
        """
        if batch_size is None:
            batch_size = self.batch_size
        idx = np.random.randint(0, x.shape[0], batch_size)
        real_traffic = x[idx]
        generated_traffic = self.generate(number=batch_size)
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))
        d_loss_real = np.mean(self.discriminator.evaluate(x=real_traffic, y=ones, verbose=False))
        d_loss_fake = np.mean(self.discriminator.evaluate(x=generated_traffic, y=zeros, verbose=False))
        d_l = 0.5 * np.add(d_loss_real, d_loss_fake)
        valid = np.ones((batch_size, 1))
        g_l = self.combined.evaluate(x=noise, y=valid, verbose=False)
        return float(d_l), float(g_l)

    def return_models(self):
        return self.generator, self.discriminator, self.combined

    def plot_learning(self):
        plt.plot(self.history["d_loss"], label="discriminator loss")
        plt.plot(self.history["g_loss"], label="generator loss")
        plt.xlabel("epochs")
        plt.title("Learning evolution")
        plt.legend()
        plt.show()
        plt.close()
        return True

    def save_model(self, location="models/", model_name="test1.0"):
        # generator
        generator_json = self.generator.to_json()
        generator_path = location + model_name + "GENERATOR"
        with open(generator_path + ".json", "w") as json_file:
            json_file.write(generator_json)
        self.generator.save_weights(generator_path + ".h5")
        print("Saved generator to disk")
        # discriminator
        discriminator_json = self.discriminator.to_json()
        discriminator_path = location + model_name + "DISCRIMINATOR"
        with open(discriminator_path + ".json", "w") as json_file:
            json_file.write(discriminator_json)
        self.discriminator.save_weights(discriminator_path + ".h5")
        print("Saved discriminator to disk")
        return True

    def load_model(self, location, model_name):
        # generator
        generator_path = location + model_name + "GENERATOR"
        generator_file = open(generator_path + ".json", 'r')
        loaded_model_json = generator_file.read()
        generator_file.close()
        self.generator = model_from_json(loaded_model_json)
        # load weights into new model
        self.generator.load_weights(generator_path + ".h5")
        if self.verbose:
            print("Loaded GENERATOR from disk")
        # discriminator
        discriminator_path = location + model_name + "DISCRIMINATOR"
        discriminator_file = open(discriminator_path + ".json", 'r')
        loaded_model_json = discriminator_file.read()
        discriminator_file.close()
        self.discriminator = model_from_json(loaded_model_json)
        # load weights into new model
        self.discriminator.load_weights(discriminator_path + ".h5")
        if self.verbose:
            print("Loaded DISCRIMINATOR from disk")

        self.discriminator.compile(loss=self.discriminator_loss,
                                   optimizer=self.optimizer)
        self.build_combined()
        if self.verbose:
            print("MODEL COMPILED")
