import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras.datasets import mnist
from keras import backend as K
from ploting import generateImages
from save_data import save_data
from testingParameters import load_GAN, trainGAN



K.set_image_dim_ordering('th')
np.random.seed(1000)
randomDim = 50

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_test = (X_test.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

size = 40000
X_train, y_train = X_train[:size], y_train[:size]

epochs = 100
alphas = [1., 2., 5., 10.]
offsets = [0., 0.1, 1.2, 2.1, 10.2]

historique_malveillance = {"alpha": dict(), "offset": dict(), "exp": dict(), "pow": dict(), "sum": dict()}

print("TESTING ALPHAS")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")

for alpha in alphas:
    generator, discriminator, GAN = load_GAN(alpha=alpha, offset=0., loss_mode="alpha")
    _, malveillance, GANloss = trainGAN(discriminator, generator, GAN, X_train=X_train,
                                        epochs=epochs,
                                        batchSize=128,
                                        dLossLimit=0.1,
                                        randomDim=randomDim, save_mode=False)
    historique_malveillance["alpha"][str(alpha)] = dict()
    historique_malveillance["alpha"][str(alpha)]["malveillance"] = malveillance
    historique_malveillance["alpha"][str(alpha)]["GANloss"] = GANloss
    historique_malveillance["alpha"][str(alpha)]["images"] = generateImages(generator=generator,
                                                                            randomDim=randomDim,
                                                                            examples=100)
    save_data(dico=historique_malveillance["alpha"][str(alpha)],
              title="alpha "+str(alpha),
              save_name="alpha"+str(alpha),
              generator=generator, X_train=X_train, randomDim=randomDim)


print("TESTING OFFSETS")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")

for offset in offsets:
    generator, discriminator, GAN = load_GAN(alpha=1., offset=offset)
    _, malveillance, GANloss = trainGAN(discriminator, generator, GAN, X_train=X_train,
                                        epochs=epochs,
                                        batchSize=128,
                                        dLossLimit=0.1,
                                        randomDim=randomDim, save_mode=True)
    historique_malveillance["offset"][str(offset)] = dict()
    historique_malveillance["offset"][str(offset)]["malveillance"] = malveillance
    historique_malveillance["offset"][str(offset)]["GANloss"] = GANloss
    historique_malveillance["offset"][str(offset)]["images"] = generateImages(generator=generator,
                                                                              randomDim=randomDim,
                                                                              examples=100)
    save_data(dico=historique_malveillance["offset"][str(offset)],
              title="INVERToffset "+str(offset),
              save_name="INVERToffset"+str(offset),
              generator=generator, X_train=X_train, randomDim=randomDim)


print("TESTING POWERS")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
power = 3
generator_excess, discriminator_excess, GAN_excess = load_GAN(loss_mode="pow", power=power)
to_be_used_excess, malveillance_excess, GANloss_excess = trainGAN(discriminator_excess, generator_excess, GAN_excess,
                                                                  X_train=X_train,
                                                                  epochs=epochs,
                                                                  batchSize=128,
                                                                  dLossLimit=0.1,
                                                                  randomDim=randomDim)
historique_malveillance["pow"][str(power)] = dict()
historique_malveillance["pow"][str(power)]["malveillance"] = malveillance_excess
historique_malveillance["pow"][str(power)]["GANloss"] = GANloss_excess
historique_malveillance["pow"][str(power)]["images"] = generateImages(generator=generator_excess,
                                                                      randomDim=randomDim,
                                                                      examples=100)
save_data(dico=historique_malveillance["pow"][str(power)],
          title="pow "+str(power),
          save_name="power"+str(power),
          generator=generator_excess, X_train=X_train, randomDim=randomDim)


print("TESTING EXP")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")

generator_excess, discriminator_excess, GAN_excess = load_GAN(loss_mode="exp")
_, malveillance_excess, GANloss_excess = trainGAN(discriminator_excess, generator_excess, GAN_excess,
                                                  X_train=X_train,
                                                  epochs=epochs,
                                                  batchSize=128,
                                                  dLossLimit=0.1,
                                                  randomDim=randomDim)

historique_malveillance["exp"]["1"] = dict()
historique_malveillance["exp"]["1"]["malveillance"] = malveillance_excess
historique_malveillance["exp"]["1"]["GANloss"] = GANloss_excess
historique_malveillance["exp"]["1"]["images"] = generateImages(generator=generator_excess,
                                                               randomDim=randomDim,
                                                               examples=100)
save_data(dico=historique_malveillance["exp"]["1"],
          title="exp",
          save_name="exp",
          generator=generator_excess, X_train=X_train, randomDim=randomDim)


print("TESTING SUM")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")

generator_sum, discriminator_sum, GAN_sum = load_GAN(loss_mode="sum", offset=0.)
to_be_used_sum, malveillance_sum, GANloss_sum = trainGAN(discriminator_sum, generator_sum, GAN_sum, X_train=X_train,
                                                         epochs=epochs,
                                                         batchSize=128,
                                                         dLossLimit=0.1,
                                                         randomDim=randomDim)


historique_malveillance["sum"]["1"] = dict()
historique_malveillance["sum"]["1"]["malveillance"] = malveillance_sum
historique_malveillance["sum"]["1"]["GANloss"] = GANloss_sum
historique_malveillance["sum"]["1"]["images"] = generateImages(generator=generator_sum,
                                                               randomDim=randomDim,
                                                               examples=100)


save_data(dico=historique_malveillance["sum"]["1"],
          title="sum",
          save_name="sum",
          generator=generator_sum, X_train=X_train, randomDim=randomDim)

print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
