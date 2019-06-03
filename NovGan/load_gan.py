from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
from losses.losses import custom_loss, custom_loss_discriminator


def load_gan(offset=0., alpha=1, randomDim=50, link_mode="alpha", power=1, mult=1, sqrt=1, loss_base="Goodfellow"):
    assert link_mode in ["alpha", "exp", "pow", "sum"], \
        "Loss function: " + link_mode + " not supported, please use alpha, exp, pow"
    assert loss_base in ["Goodfellow", "Wasserstein", "Pearson"], "This loss: " + loss_base + " is not supported"

    adam = Adam(lr=0.0002, beta_1=0.5)

    generator = Sequential()
    generator.add(Dense(256, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(784, activation='tanh'))
    generator.compile(loss="binary_crossentropy", optimizer=adam)

    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator_loss = custom_loss_discriminator(loss_base=loss_base)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)

    # Combined network
    discriminator.trainable = False
    ganInput = Input(shape=(randomDim,))
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

    gan.compile(loss='binary_crossentropy', optimizer=adam)

    return generator, discriminator, gan
