from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import initializers
from keras.models import model_from_json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import numpy as np
from evaluation.evaluation import evaluate

place = "home"
import sys
if place == "home":
    sys_path = "/Users/ppx/Desktop/gitELECOM/spectralNormalisation"
elif place == "work":
    sys_path = "/home/peseux/Desktop/gitELECOM/spectralNormalisation/"
sys.path.insert(0, sys_path)
from dense_spectral_normalisation import DenseSN

if place == "work":
    sys_path = "/home/peseux/Desktop/gitELECOM/cGANoDEbergerac/loadingCGAN"
elif place == "home":
    sys_path = "/Users/ppx/Desktop/gitELECOM/cGANoDEbergerac/loadingCGAN"
sys.path.insert(0, sys_path)
from weight_clipping import WeightClip
from utils_cgan import false_or_true, proba_choice, past_labeling, tanh_to_zero_one, smoothing_y


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
        list_of_gans[i].discriminator = discriminators[sigma[i]]
        list_of_gans[i].build_combined()
    print("GANs switched")
    return list_of_gans, sigma


class Cgan(object):
    def __init__(self, data_dim=28, num_classes=2,
                 latent_dim=32, batch_size=128, leaky_relu=.2,
                 dropout=.4, spectral_normalisation=False,
                 weight_clipping=False,
                 weight_clip=1,
                 verbose=False,
                 activation="sigmoid",
                 gan_loss="binary_crossentropy",
                 discriminator_loss="binary_crossentropy",
                 noise="normal"):
        # Input shape
        self.data_dim = data_dim
        self.num_classes = num_classes
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

        self.past_images = dict()
        for i in range(self.num_classes):
            self.past_images[str(i)] = self.generate(number=self.batch_size, labels=np.full((self.batch_size, 1), i))

        self.history = {"cv_loss": [], "d_loss": [], "g_loss": []}

    def build_generator(self):
        if self.spectral_normalisation:
            dense = DenseSN
        else:
            dense = Dense
        if self.weight_clipping:
            kernel_constraint = WeightClip(self.weight_clip)
        else:
            kernel_constraint = None

        model = Sequential()

        model.add(dense(12, input_dim=self.latent_dim,
                        kernel_initializer=initializers.RandomNormal(stddev=0.02),
                        W_constraint=kernel_constraint))
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(Dropout(self.dropout))
        model.add(dense(64,
                        kernel_initializer=initializers.RandomNormal(stddev=0.02),
                        W_constraint=kernel_constraint))
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(Dropout(self.dropout))
        model.add(dense(self.data_dim,
                        kernel_initializer=initializers.RandomNormal(stddev=0.02),
                        W_constraint=kernel_constraint))
        if self.verbose:
            print("\n \n Generator Architecture ")
            model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):
        if self.spectral_normalisation:
            dense = DenseSN
        else:
            dense = Dense
        if self.weight_clipping:
            kernel_constraint = WeightClip(self.weight_clip)
        else:
            kernel_constraint = None

        model = Sequential()

        model.add(dense(18, input_dim=np.prod(self.data_dim), kernel_constraint=kernel_constraint))
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(dense(12, kernel_constraint=kernel_constraint))
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(Dropout(self.dropout))
        model.add(dense(10, kernel_constraint=kernel_constraint))
        model.add(LeakyReLU(alpha=self.leaky_relu))
        model.add(Dropout(self.dropout))
        model.add(dense(1,
                        kernel_constraint=kernel_constraint,
                        activation=self.activation))
        if self.activation == "tanh":
            model.add(dense(1, activation=tanh_to_zero_one))

        traffic = Input(shape=(self.data_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.data_dim)(label))
        model_input = multiply([traffic, label_embedding])
        validity = model(model_input)
        if self.verbose:
            print("\n \n Discriminator Architecture ")
            model.summary()
        return Model([traffic, label], validity)

    def build_combined(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        traffic = self.generator([noise, label])
        valid = self.discriminator([traffic, label])
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=self.gan_los,
                              optimizer=self.optimizer)

    def generate(self, number, labels):

        if self.noise == "normal":
            noise = np.random.normal(0, 1, (number, self.latent_dim))
        elif self.noise == "logistic":
            noise = np.random.logistic(0, 1, (number, self.latent_dim))
        generated_traffic = self.generator.predict([noise, labels])
        return generated_traffic

    def train(self, x_train, y_train, epochs, cv_size=.2, print_recap=True,
              reload_images_p=.8, show_past_p=.9, smooth_zero=.1, smooth_one=.9):
        """

        :param x_train:
        :param y_train:
        :param epochs:
        :param cv_size:
        :param print_recap:
        :param reload_images_p:
        :param show_past_p:
        :param smooth_zero:
        :param smooth_one:
        :return:
        """
        cv_loss, d_loss, g_loss = list(), list(), list()
        x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(x_train, y_train, test_size=cv_size)
        ones = np.ones((x_test_cv.shape[0], 1))
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        batch_count = int(x_train.shape[0] / self.batch_size)

        for _ in tqdm(range(epochs)):
            cv_l, d_l, g_l = 0., 0., 0.
            for _ in (range(batch_count)):
                # Reload past images
                if np.random.random() > reload_images_p:
                    for i in range(self.num_classes):
                        self.past_images[str(i)] = self.generate(number=self.batch_size,
                                                                 labels=np.full((self.batch_size, 1), i))

                #  Train Discriminator
                # Select a random half batch of images
                idx = np.random.randint(0, x_train_cv.shape[0], self.batch_size)
                real_traffic, labels = x_train_cv[idx], y_train_cv[idx]
                if np.random.random() > show_past_p:
                    generated_traffic = past_labeling(traffics=self.past_images,
                                                      lab=labels)
                else:
                    generated_traffic = self.generate(number=self.batch_size, labels=labels)
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch([real_traffic, labels], smoothing_y(valid,
                                                                                                    smooth_zero=smooth_zero,
                                                                                                    smooth_one=smooth_one))
                d_loss_fake = self.discriminator.train_on_batch([generated_traffic, labels], fake)
                d_l += 0.5 * np.add(d_loss_real, d_loss_fake)
                self.discriminator.trainable = False
                #  Train Generator
                # Condition on labels
                sampled_labels = np.random.randint(0, self.num_classes, self.batch_size).reshape(-1, 1)
                g_l += self.combined.train_on_batch([noise,sampled_labels], valid)
                cv_l += np.mean(self.discriminator.evaluate(x=[x_test_cv, y_test_cv], y=ones, verbose=False))

            cv_loss.append(cv_l/batch_count)
            d_loss.append(d_l/batch_count)
            g_loss.append(g_l/batch_count)
        self.history["cv_loss"] = self.history["cv_loss"] + cv_loss
        self.history["d_loss"] = self.history["d_loss"] + d_loss
        self.history["g_loss"] = self.history["g_loss"] + g_loss
        if print_recap:
            self.plot_learning()
        return cv_loss, d_loss, g_loss

    def evaluate(self, x, y, batch_size=None, mode_d_loss=True):
        """
        :param mode_d_loss:
        :param x:
        :param y:
        :param batch_size:
        :return: d_l, g_l
        """
        if batch_size is None:
            batch_size = self.batch_size
        idx = np.random.randint(0, x.shape[0], batch_size)
        real_traffic, labels = x[idx], y[idx]
        generated_traffic = self.generate(number=batch_size, labels=labels)
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))
        if mode_d_loss:
            d_loss_real = np.mean(self.discriminator.evaluate(x=[real_traffic, labels], y=ones))
            d_loss_fake = np.mean(self.discriminator.evaluate(x=[generated_traffic, labels], y=zeros))
            d_l = 0.5 * np.add(d_loss_real, d_loss_fake)
        else:
            y_pred_one = self.discriminator.predict(x=[real_traffic, labels])
            y_pred_zero = self.discriminator.predict(x=[generated_traffic, labels])
            y_true = np.concatenate([ones, zeros])
            y_pred = [int(y) for y in np.concatenate([y_pred_one, y_pred_zero])]
            val = evaluate(y_true=y_true, y_pred=y_pred)
            d_l = - val["f1_score"]

        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        valid = np.ones((batch_size, 1))
        g_l = self.combined.evaluate(x=[noise, sampled_labels], y=valid)
        return float(d_l), float(g_l)

    def return_models(self):
        return self.generator, self.discriminator, self.combined

    def predict(self, x, threshold=.5):
        y_pred = self.predict_proba(x)
        y_pred = [int(y > threshold) for y in y_pred]
        return np.array(y_pred)

    def predict_proba(self, x):
        ones = np.ones((x.shape[0], 1))
        zeros = np.zeros((x.shape[0], 1))
        y_pred_ones = self.discriminator.predict([x, ones])
        y_pred_zeros = self.discriminator.predict([x, zeros])
        y_proba = [proba_choice([y0[0], y1[0]]) for y0, y1 in zip(y_pred_zeros, y_pred_ones)]
        return np.array(y_proba)

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
