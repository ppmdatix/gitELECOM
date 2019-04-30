from ploting import generateImages
from matplotlib import pyplot as plt
import numpy as np


def save_histogram_malveillance(generator, X_train, save_name="image", randomDim=50, examples=10000):
    images = generateImages(generator=generator, randomDim=randomDim, examples=examples)
    plt.hist([np.mean(x) for x in X_train],bins=50, label="real")
    plt.hist([np.mean(x) for x in images],bins=100, label="generated")
    plt.legend
    plt.savefig("images" + save_name + ".png")
    plt.close()

    return True


