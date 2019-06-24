from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
from utils_cgan import tanh_to_zero_one


def zero_or_one(x):
    if x < .5:
        return 0
    else:
        return 1


class Mlp(object):
    def __init__(self, data_dim=28, num_classes=2, activation="tanh"):
        # Input shape
        self.data_dim = data_dim
        self.num_classes = num_classes
        self.activation = activation

        optimizer = Adam(0.0002, 0.5)

        print("CHOSEN OPTIMIZER IS ADAM")

        # Build and compile the discriminator
        self.mlp = self.build_mlp()
        self.mlp.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=['accuracy'])

    def build_mlp(self):

        model = Sequential()

        model.add(Dense(18, input_dim=self.data_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(12))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation=self.activation))
        if self.activation == "tanh":
            model.add(Dense(1, activation=tanh_to_zero_one))
        model.summary()

        return model

    def train(self, x_train, y_train, epochs, batch_size=128):
        """

        :param x_train:
        :param y_train:
        :param epochs:
        :param batch_size:
        :return: d_loss
        """

        batch_count = int(x_train.shape[0] / batch_size)
        for epoch in tqdm(range(epochs)):
            for _ in range(batch_count):
                # Select a random half batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_traffic, labels = x_train[idx], y_train[idx]
                # Train the discriminator
                d_loss = self.mlp.train_on_batch(real_traffic, labels)
        return d_loss

    def return_models(self):
        return self.mlp

    def predict(self, x):
        y_pred = self.mlp.predict(x)
        y_p = [zero_or_one(y) for y in y_pred]
        return y_p

