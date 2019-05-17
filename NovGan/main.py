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

size = 60000
X_train, y_train = X_train[:size], y_train[:size]

epochs = 40
alphas = [1., 2., 5., 10.]
offsets = [0., 0.1, 1.2, 2.1, 10.2]
powers = [2, 5, 10]

dico_link = {"alpha": [{"alpha": 1.,
                        "offset": 0,
                        "malveillance": None,
                        "GANloss": None,
                        "images": None
                        }],
             "exp": [{"malveillance": None,
                      "GANloss": None,
                      "images": None
                      }],
             "pow": [{"power": 1.,
                      "malveillance": None,
                      "GANloss": None,
                      "images": None
                      }],
             "sum": [{"mult": 1.,
                      "sqrt": 1.,
                      "malveillance": None,
                      "GANloss": None,
                      "images": None
                      }]}

historique = {"Goodfellow": dico_link,
              "Wasserstein": dico_link,
              "Pearson": dico_link
              }

print("TESTING ALPHAS")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")
print("=====================================================================")

for alpha, offset in zip(alphas, offsets):
    #==================#
    #    Goodfellow    #
    #==================#
    loss_base = "Goodfellow"
    link_mode = "alpha"

    generator, discriminator, GAN = load_GAN(alpha=alpha,
                                             offset=0.,
                                             link_mode=link_mode,
                                             loss_base=loss_base)
    _, malveillance, GANloss = trainGAN(discriminator, generator, GAN, X_train=X_train,
                                        epochs=epochs,
                                        batchSize=128,
                                        dLossLimit=0.1,
                                        randomDim=randomDim,
                                        save_mode=True,
                                        save_title=loss_base+"alpha" + str(alpha) + "and_offset" + str(offset))
    g_images = generateImages(generator=generator,
                              randomDim=randomDim,
                              examples=100)
    dic = {"alpha": alpha,
           "offset": offset,
           "malveillance": malveillance,
           "GANloss": GANloss,
           "images": g_images
           }

    historique[loss_base][link_mode].append(dic)
    save_data(dico=dic,
              title="alpha " + str(alpha) + "and offset " + str(offset),
              save_name="alpha" + str(alpha) + "and_offset" + str(offset),
              generator=generator, X_train=X_train, randomDim=randomDim)


    #==================#
    #    Wasserstein   #
    #==================#
    loss_base = "Wasserstein"
    link_mode = "alpha"

    generator, discriminator, GAN = load_GAN(alpha=alpha,
                                             offset=0.,
                                             link_mode=link_mode,
                                             loss_base=loss_base)
    _, malveillance, GANloss = trainGAN(discriminator, generator, GAN, X_train=X_train,
                                        epochs=epochs,
                                        batchSize=128,
                                        dLossLimit=0.1,
                                        randomDim=randomDim,
                                        save_mode=True,
                                        save_title=loss_base+"alpha" + str(alpha) + "and_offset" + str(offset))
    g_images = generateImages(generator=generator,
                              randomDim=randomDim,
                              examples=100)
    dic = {"alpha": alpha,
           "offset": offset,
           "malveillance": malveillance,
           "GANloss": GANloss,
           "images": g_images
           }

    historique[loss_base][link_mode].append(dic)
    save_data(dico=dic,
              title="alpha " + str(alpha) + "and offset " + str(offset),
              save_name=loss_base+ "/" +"alpha" + str(alpha) + "and_offset" + str(offset),
              generator=generator, X_train=X_train, randomDim=randomDim)

    #==================#
    #    Pearson   #
    #==================#
    loss_base = "Pearson"
    link_mode = "alpha"

    generator, discriminator, GAN = load_GAN(alpha=alpha,
                                             offset=0.,
                                             link_mode=link_mode,
                                             loss_base=loss_base)
    _, malveillance, GANloss = trainGAN(discriminator, generator, GAN, X_train=X_train,
                                        epochs=epochs,
                                        batchSize=128,
                                        dLossLimit=0.1,
                                        randomDim=randomDim,
                                        save_mode=True,
                                        save_title=loss_base+"alpha" + str(alpha) + "and_offset" + str(offset))
    g_images = generateImages(generator=generator,
                              randomDim=randomDim,
                              examples=100)
    dic = {"alpha": alpha,
           "offset": offset,
           "malveillance": malveillance,
           "GANloss": GANloss,
           "images": g_images
           }

    historique[loss_base][link_mode].append(dic)
    save_data(dico=dic,
              title="alpha " + str(alpha) + "and offset " + str(offset),
              save_name=loss_base+ "/" +"alpha" + str(alpha) + "and_offset" + str(offset),
              generator=generator, X_train=X_train, randomDim=randomDim)


print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
print("LETS GO TO SLEEP")
