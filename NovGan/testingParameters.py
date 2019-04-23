from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
import tensorflow as tf

# Custom Loss
def customLoss(layer_weights, lamda=0.5, y_func="pow10"):
    def lossFunction(y_true,y_pred):
        y = - K.log(y_pred)
        x =  (K.mean(K.abs(( - layer_weights + 1)/2)))#/2)) + 0.5*K.std(layer_weights)) / 2
        if y_func == "carre":
            fy = y*y
        elif y_func == "pow10":
            fy = y**10
        else:
            fy = y
        #loss = x * K.log( - K.abs(y_pred - y_true)) + (1 - x)*(K.log( - K.abs(y_pred - y_true))*K.log( - K.abs(y_pred - y_true)))
        loss = y + lamda * x 
        return loss

    return lossFunction
def customLossAcceleration(layer_weights, alpha=.1, offset=5):
    def lossFunction(y_true,y_pred):
        y = - K.log(y_pred)
        x = K.mean(K.abs(( - layer_weights + 1)/2))
        loss = x * y  + (1-x) * ((1/alpha)*y + offset)
        return loss

    return lossFunction

def load_GAN(offset, alpha, randomDim=50):


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
	# generator.compile(loss='binary_crossentropy', optimizer=adam)

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
	gan.compile(loss=customLossAcceleration(x, offset=offset, alpha=alpha), optimizer =adam)
	#gan.compile(loss=customLoss(x, lamda), optimizer =adam)

	return generator, discriminator, gan

