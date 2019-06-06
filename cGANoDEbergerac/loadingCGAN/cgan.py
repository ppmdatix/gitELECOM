from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import initializers
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import numpy as np


def zero_or_one(x):
    if x < .5:
        return 0
    else:
        return 1


def false_or_true(x):
    if x[0] < .5 and x[1] < .5:
        return 0
    elif x[0] > x[1]:
        return 0
    elif x[0] == x[1]:
        print("FUCK")
        return 1
    else:
        return 1


def proba_choice(x):
    if x[0] > x[1]:
        return 1 - x[0]
    elif x[0] == x[1]:
        print("FUCK")
        return 1
    else:
        return x[1]


def past_labeling(traffics, lab):
    output = []
    size = len(lab)
    for i in range(size):
        label = lab[i]
        output.append(traffics[str(int(label))][i])
    return np.array(output)


def switching_gans(list_of_gans):
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
        list_of_gans[i].generator = generators[sigma[i]]
        list_of_gans[i].discriminator = discriminators[sigma[i]]
        list_of_gans[i].build_combined()
    print("GANs switched")
    return True


class Cgan(object):
    def __init__(self, data_dim=28, num_classes=2,
                 latent_dim=32, batch_size=128, leaky_relu=.02, dropout=.4):
        # Input shape
        self.data_dim = data_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.optimizer = Adam(0.0002, 0.5)
        print("CHOSEN OPTIMIZER IS ADAM")
        self.leaky_relu = leaky_relu
        self.dropout = dropout
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.discriminator.trainable = False
        self.combined = None
        self.build_combined()

        self.past_images = dict()
        for i in range(self.num_classes):
            self.past_images[str(i)] = self.generate(number=self.batch_size, labels=np.full((self.batch_size, 1), i))

        self.history = {"cv_loss": [], "d_loss": [], "g_loss": []}

    def build_generator(self):

        model = Sequential()

        model.add(Dense(12, input_dim=self.latent_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(18))
        model.add(LeakyReLU(alpha=self.leaky_relu))
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
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(Dense(12))
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(Dropout(self.dropout))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='sigmoid'))

        traffic = Input(shape=(self.data_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.data_dim)(label))
        model_input = multiply([traffic, label_embedding])
        validity = model(model_input)
        model.summary()
        return Model([traffic, label], validity)

    def build_combined(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        traffic = self.generator([noise, label])
        valid = self.discriminator([traffic, label])
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=self.optimizer)

    def generate(self, number, labels):
        noise = np.random.normal(0, 1, (number, self.latent_dim))
        generated_traffic = self.generator.predict([noise, labels])
        return generated_traffic

    def train(self, x_train, y_train, epochs,
              batch_size=128, cv_size=.2, print_recap=True,
              reload_images_p=.8, show_past_p=.9):
        """
        :param x_train:
        :param y_train:
        :param epochs:
        :param batch_size:
        :param cv_size:
        :param print_recap:
        :param reload_images_p:
        :param show_past_p:
        :return: cv_loss, d_loss, g_loss
        """
        self.batch_size = batch_size
        cv_loss, d_loss, g_loss = list(), list(), list()
        x_trainCV, x_testCV, y_trainCV, y_testCV = train_test_split(x_train, y_train, test_size=cv_size)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for _ in tqdm(range(epochs)):
            # Reload past images
            if np.random.random() > reload_images_p:
                for i in range(self.num_classes):
                    self.past_images[str(i)] = self.generate(number=self.batch_size,
                                                             labels=np.full((self.batch_size, 1), i))

            #  Train Discriminator
            # Select a random half batch of images
            idx = np.random.randint(0, x_trainCV.shape[0], batch_size)
            real_traffic, labels = x_trainCV[idx], y_trainCV[idx]
            if np.random.random() > show_past_p:
                generated_traffic = past_labeling(traffics=self.past_images,
                                                  lab=labels)
            else:
                generated_traffic = self.generate(number=batch_size, labels=labels)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            d_loss_real = self.discriminator.train_on_batch([real_traffic, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([generated_traffic, labels], fake)
            d_l = 0.5 * np.add(d_loss_real, d_loss_fake)
            #  Train Generator
            # Condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            g_l = self.combined.train_on_batch([noise, sampled_labels], valid)
            ones = np.ones((x_testCV.shape[0], 1))
            cv_l = np.mean(self.discriminator.evaluate(x=[x_testCV, y_testCV], y=ones, verbose=False))

            cv_loss.append(cv_l)
            d_loss.append(d_l)
            g_loss.append(g_l)
        if print_recap:
            plt.figure(figsize=(10, 5))
            plt.plot(cv_loss, label="CV SCORE")
            plt.plot(d_loss, label="discriminator loss")
            plt.plot(g_loss, label="generator loss")
            plt.legend()
            plt.show()
            plt.close()
        self.history["cv_loss"] = self.history["cv_loss"] + cv_loss
        self.history["d_loss"] = self.history["d_loss"] + d_loss
        self.history["g_loss"] = self.history["g_loss"] + g_loss
        return cv_loss, d_loss, g_loss

    def return_models(self):
        return self.generator, self.discriminator, self.combined

    def predict(self, x):
        ones = np.ones((x.shape[0], 1))
        zeros = np.zeros((x.shape[0], 1))
        y_pred_ones = self.discriminator.predict([x, ones])
        y_pred_zeros = self.discriminator.predict([x, zeros])
        y_pred = [false_or_true([y0[0], y1[0]]) for y0, y1 in zip(y_pred_zeros, y_pred_ones)]
        return np.array(y_pred)

    def predict_proba(self, x):
        ones = np.ones((x.shape[0], 1))
        zeros = np.zeros((x.shape[0], 1))
        y_pred_ones = self.discriminator.predict([x, ones])
        y_pred_zeros = self.discriminator.predict([x, zeros])
        y_proba = [proba_choice([y0[0], y1[0]]) for y0, y1 in zip(y_pred_zeros, y_pred_ones)]
        return np.array(y_proba)

    def plot_learning(self):
        plt.plot(self.history["cv_loss"], label="cv_loss")
        plt.plot(self.history["d_loss"], label="discriminator loss")
        plt.plot(self.history["g_loss"], label="generator loss")
        plt.legend()
        plt.show()
        plt.close()
        return True





