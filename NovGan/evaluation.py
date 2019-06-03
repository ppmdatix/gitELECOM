from matplotlib import pyplot as plt
from train_gan import generateImages
import numpy as np


def plotImages(generatedImages, dim=(10,10), title="title", location="test.png", save_mode=False):
    plt.figure(figsize=dim)
    plt.title(title)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if save_mode:
        plt.savefig(location + "IMAGES.png")
    plt.close()


def plot_learning(hurting, gen_loss, disc_loss, location):
    plt.plot(hurting, label="malveillance")
    plt.plot(gen_loss, label="generator loss")
    plt.plot(disc_loss, label="disc loss")
    plt.legend()
    plt.savefig(location + "LEARNING.png")
    plt.close()
    return True


def plot_hist(generatedImages, x_test, location):
    plt.hist([np.mean(x) for x in x_test], bins=30, label="real")
    plt.hist([np.mean(x) for x in generatedImages], bins=30, label="generated")
    plt.legend()
    plt.savefig(location + "HISTO.png")
    plt.close()
    return True


def evaluation(generator, randomDim, location, disc_loss, gen_loss, hurting, x_test,
               examples=100, title="title"):
    generated_images = generateImages(generator=generator,
                                     randomDim=randomDim,
                                     examples=examples)
    plotImages(generated_images, dim=(10, 10), title=title,
               location=location, save_mode=True)

    plot_learning(hurting=hurting, gen_loss=gen_loss,
                  disc_loss=disc_loss, location=location)

    generated_images = generateImages(generator=generator,
                                     randomDim=randomDim,
                                     examples=examples*examples)

    plot_hist(generatedImages=generated_images,
              x_test=x_test,
              location=location)
    return True
