from matplotlib import pyplot as plt
from train_gan import generateImages


def plotImages(generatedImages, dim=(10,10), title="title", location="test.png", save_mode=False):
    plt.figure(figsize=dim)
    plt.title(title)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if save_mode:
        plt.savefig(location)
    plt.close()


def evaluation(generator, randomDim, examples=100, title="title"):
    generatedImages = generateImages(generator=generator,
                                     randomDim=randomDim,
                                     examples=examples)
    plotImages(generatedImages, dim=(10,10), title=title, location=location, save_mode=True)