from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Embedding
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


def false_or_true(x):
    if x[0] > x[1]:
        return 0
    elif [0] > x[1]:
        print("FUCK")
        return 1
    else:
        return 1


class Cgan(object):
    def __init__(self, data_dim=28, num_classes=2, latent_dim=32):
        # Input shape
        self.data_dim = data_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        optimizer = Adam(0.0002, 0.5)

        print("CHOSEN OPTIMIZER IS ADMA")

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        traffic = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([traffic, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(12, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(18))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.data_dim, activation='tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(18, input_dim=np.prod(self.data_dim)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(12))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        traffic = Input(shape=(self.data_dim,))
        label = Input(shape=(1,), dtype='int32')


        label_embedding = Flatten()(Embedding(self.num_classes, self.data_dim)(label))
        #flat_traffic = Flatten()(traffic)

        model_input = multiply([traffic, label_embedding])

        validity = model(model_input)

        return Model([traffic, label], validity)

    def train(self, x_train, y_train, epochs, batch_size=128, sample_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in tqdm(range(epochs)):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_traffic, labels = x_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            generated_traffic = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([real_traffic, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([generated_traffic, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    def return_models(self):
        return self.generator, self.discriminator, self.combined

    def evaluate_discriminator(self, x_test, y_test):
        ones = np.ones((x_test.shape[0], 1))
        zeros = np.zeros((x_test.shape[0], 1))
        y_pred_ones = self.discriminator.predict([x_test, ones])
        y_pred_zeros = self.discriminator.predict([x_test, zeros])
        y_pred = [false_or_true([y0[0], y1[0]]) for y0, y1 in zip(y_pred_zeros, y_pred_ones)]
        conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
        print(sum(y_pred) / len(y_pred))
        return conf_matrix

