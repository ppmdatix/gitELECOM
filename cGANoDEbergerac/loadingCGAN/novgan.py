from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from keras.initializers import glorot_uniform
import sys
sys.path.append("..")
from losses.losses_novgan import custom_loss



def zero_or_one(x):
    if x < .5:
        return 0
    else:
        return 1

class Novgan(object):
    def __init__(self, data_dim=28, activation="tanh", verbose=True,
                 latent_dim=32,
                 leaky_relu=.1, offset=0, alpha=0, dropout=.2,
                 dico_index=None):
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
        self.custom_loss = custom_loss

        self.optimizer = Adam(0.0002, 0.5)
        if self.verbose:
            print("CHOSEN OPTIMIZER IS ADAM")

        # Build and compile the discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_combined()


    def build_generator(self):
        generator = Sequential()
        generator.add(Dense(64, input_dim=self.latent_dim, kernel_initializer=glorot_uniform()))
        generator.add(LeakyReLU(self.leaky_rely))
        generator.add(Dense(128))
        generator.add(LeakyReLU(self.leaky_rely))
        generator.add(Dense(self.data_dim))
        # generator.compile(loss="binary_crossentropy", optimizer=self.optimizer)

        if self.verbose:
            print("\n \n Generator Architecture ")
            generator.summary()
        return generator

    def build_discriminator(self):
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
        discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return discriminator

    def build_combined(self):

        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss=custom_loss(intermediate_output=x, alpha=self.alpha, offset=self.offset, dico=self.dico_index),
                    optimizer=self.optimizer)

        return gan



    def train(self, x_train, y_train, epochs, batch_size=128):
        """

        :param x_train:
        :param y_train:
        :param epochs:
        :param batch_size:
        :return: d_loss
        """

        batch_count = int(x_train.shape[0] / batch_size)
        for _ in range(epochs):
            for _ in range(batch_count):
                # Select a random half batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_traffic, labels = x_train[idx], y_train[idx]
                # Train the discriminator
                d_loss = self.mlp.train_on_batch(real_traffic, labels)
        return d_loss

