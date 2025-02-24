from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
from keras.initializers import glorot_uniform
import sys
# sys.path.insert(0, '/home/peseux/Desktop/gitELECOM/IDSGAN/losses/')
sys.path.insert(0, '/Users/ppx/Desktop/gitELECOM/IDSGAN/losses/')

from gan_loss import custom_loss

def load_gan(data_dim, random_dim=50):

    adam = Adam(lr=0.0002, beta_1=0.5)
    print("Chosen Optimizer is ADAM")

    generator = Sequential()
    generator.add(Dense(12, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(18))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(data_dim, activation='tanh'))
    generator.compile(loss="binary_crossentropy", optimizer=adam)
    # generator.compile(loss='binary_crossentropy', optimizer=adam)

    discriminator = Sequential()
    discriminator.add(Dense(18, input_dim=data_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(12))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(10))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)

    # Combined network
    discriminator.trainable = False
    ganInput = Input(shape=(random_dim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    # gan.compile(loss=customLossAcceleration(x, offset=offset, alpha=alpha), optimizer =adam)
    # gan.compile(loss=customLoss(x, lamda), optimizer =adam)
    gan.compile(loss="binary_crossentropy", optimizer=adam)

    return generator, discriminator, gan


def load_gan_kdd(data_dim, random_dim=32, slope_relu=.1,
                 offset=0., alpha=1, link_mode="alpha",
                 power=1, mult=1, sqrt=1, loss_base="Goodfellow"):
    assert link_mode in ["alpha", "exp", "pow", "sum", "hurting"], "Loss function not supported, please use alpha, exp or pow"
    assert loss_base in ["Goodfellow", "Wasserstein", "Pearson"], "This loss is not supported"
    adam = Adam(lr=0.0002, beta_1=0.5)
    print("Chosen Optimizer is ADAM")
    generator = Sequential()
    generator.add(Dense(64, input_dim=random_dim, kernel_initializer=glorot_uniform()))
    generator.add(LeakyReLU(slope_relu))
    generator.add(Dense(128))
    generator.add(LeakyReLU(slope_relu))
    generator.add(Dense(data_dim, activation="tanh"))
    generator.compile(loss="binary_crossentropy", optimizer=adam)

    discriminator = Sequential()
    discriminator.add(Dense(256, input_dim=data_dim, kernel_initializer=glorot_uniform()))
    discriminator.add(LeakyReLU(slope_relu))
    discriminator.add(Dropout(0.2))
    discriminator.add(Dense(128))
    discriminator.add(LeakyReLU(slope_relu))
    discriminator.add(Dropout(0.2))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)

    # Combined network
    discriminator.trainable = False
    ganInput = Input(shape=(random_dim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    loss = custom_loss(intermediate_output=x,
                       power=power,
                       alpha=alpha,
                       offset=offset,
                       mult=mult,
                       sqrt=sqrt,
                       loss_base=loss_base,
                       link_mode=link_mode)
    gan.compile(loss=loss, optimizer=adam)
    print("Compiled GAN")

    return generator, discriminator, gan

