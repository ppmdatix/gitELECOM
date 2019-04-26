import numpy as np
from matplotlib import pyplot as plt

# Plot the loss from each batch


def plotLoss(epoch, dLosses, gLosses):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)
    
def plotingLosses(m,f,g,d):
    plt.plot(m, label="malveillance")
    plt.plot(f, label="fooling")
    plt.plot(g, label="generator loss")
    plt.plot(d, label="disc loss")
    plt.legend()
    plt.show()
    plt.close()
    return True
    

def plotLossUNS(epoch, dLosses, gLosses):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/UNSgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images


def generateImages(generator, randomDim, examples):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)
    return generatedImages


def plotImages(generatedImages, dim=(10,10), title="title"):
    plt.figure(figsize=dim)
    plt.title(title)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()


def plotMalveillance(malveilllance, title="title"):
    plt.figure(figsize=(5,5))
    plt.plot(malveilllance)
    plt.ylim([-0.1, 2])
    plt.title(title)
    plt.show()
    plt.close()
    return True


def plotGeneratedImages(epoch, generator, randomDim, examples=100, dim=(10, 10), figsize=(10, 10), save_mode=False):
    generateImages(generator, randomDim, examples)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if save_mode:
        plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)
        plt.show()
    else:
        plt.show()
    return True
    
    
    
def plotGeneratedImagesUNS(epoch, generator, randomDim, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/UNSgan_generated_image_epoch_%d.png' % epoch)