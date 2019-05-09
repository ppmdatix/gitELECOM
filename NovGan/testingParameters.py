from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
import numpy as np
from ploting import saveImages
from ploting import generateImages


# Custom Loss
def customLossAcceleration(intermediate_output, alpha=.1, offset=5):
    def lossFunction(y_true,y_pred):
        # On veut des gros chiffres

        L = - K.log(y_pred)
        L_bis = K.mean(K.abs(( - intermediate_output + 1)/2))
        loss = L * L_bis + (1-L_bis) * (alpha*L + offset)
        return loss

    return lossFunction


def customLossExponential(intermediate_output):
    def lossFunction(y_true,y_pred):
        # On veut des gros chiffres

        L = - K.log(y_pred)
        L_bis = K.mean(K.abs(( intermediate_output + 1)/2))
        loss = L * L_bis + (1-L_bis) * K.exp(L)
        return loss

    return lossFunction


def customLossPow(intermediate_output, power):
    def lossFunction(y_true,y_pred):
        # On veut des gros chiffres

        L = - K.log(y_pred)
        L_bis = K.mean(K.abs(( intermediate_output + 1)/2))
        loss = L * L_bis + (1-L_bis) * K.pow(x=L, a=power)
        return loss

    return lossFunction


def customLossSum(intermediate_output, mult=1, sqrt=2):
    def lossFunction(y_true,y_pred):
        # On veut des gros chiffres

        L = - K.log(y_pred)
        L_bis = K.mean(K.abs(( -intermediate_output + 1)/2))
        loss = L + mult * K.pow(L_bis, float(1/sqrt))
        return loss

    return lossFunction


def load_GAN(offset=0., alpha=1, randomDim=50, loss_mode="alpha", power=1, mult=1, sqrt=1):

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
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)

    # Combined network
    discriminator.trainable = False
    ganInput = Input(shape=(randomDim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    assert loss_mode in ["alpha", "exp", "pow", "sum"], "Loss function not supported, please use alpha, exp or pow"
    if loss_mode == "alpha":
        loss = customLossAcceleration(intermediate_output=x, offset=offset, alpha=alpha)
    elif loss_mode == "exp":
        loss = customLossExponential(intermediate_output=x)
    elif loss_mode == "pow":
        loss = customLossPow(intermediate_output=x, power=power)
    elif loss_mode == "sum":
        loss = customLossSum(intermediate_output=x, mult=mult, sqrt=sqrt)


    gan.compile(loss=loss, optimizer=adam)

    return generator, discriminator, gan


def malveillance(image):
    return np.mean(np.abs(image + np.ones(784))/2)


def evaluateGeneratedImagesMalveillance(generator, randomDim, examples=10):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    result = [malveillance(x) for x in generatedImages]

    return np.mean(result)


def trainGAN(disc, gen, GAN, X_train,
             epochs=1,
             batchSize=128,
             dLossLimit=0.1,
             randomDim=50,
             save_mode=False):
    dL = []
    gL = []
    mL = []

    toBeTrusted = True
    batchCount = int(X_train.shape[0] / batchSize)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        dloss, gloss = 0., 0.
        for _ in (range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = gen.predict(noise)

            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            disc.trainable = True
            dloss = disc.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            disc.trainable = False
            gloss = GAN.train_on_batch(noise, yGen)

        if dloss < dLossLimit:
            toBeTrusted = False
            break

        if save_mode and (e == 1 or e % 20 == 0):
            title, save_name = str(e), "tmp/" + str(e)
            saveImages(generateImages(generator=gen, randomDim=randomDim, examples=100).reshape(100, 28, 28), title=title, save_name=save_name)
        # Store loss of most recent batch from this epoch
        dL.append(dloss)
        gL.append(gloss)

        # Specific Loss M
        mL.append(evaluateGeneratedImagesMalveillance(generator=gen, randomDim=randomDim, examples=10))

    if toBeTrusted:
        return True, mL, gL
    else:
        print("=========ERROR=========")
        print("The Disc-Loss goesssssssssss friend")
        return False, [], []