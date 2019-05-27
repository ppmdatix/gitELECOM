from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as confusion_matrix
import numpy as np


def zero_or_one(x):
    if x < .5:
        return 0
    else:
        return 1


class Mlp(object):
    def __init__(self, data_dim=28, num_classes=2):
        # Input shape
        self.data_dim = data_dim
        self.num_classes = num_classes

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
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model

    def train(self, x_train, y_train, epochs, batch_size=128):

        for epoch in tqdm(range(epochs)):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_traffic, labels = x_train[idx], y_train[idx]
            # Train the discriminator
            d_loss = self.mlp.train_on_batch(real_traffic, labels)

    def return_models(self):
        return self.mlp

    def predict(self, x):
        y_pred = self.mlp.predict(x)
        y_p = [zero_or_one(y) for y in y_pred]
        return y_p

