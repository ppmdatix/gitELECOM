from loadingCGAN.swagan import switching_swagans
import numpy as np
from matplotlib import pyplot as plt

def plot_images(generatedImages, dim=(10,10), title="title", save_mode=False):
    plt.figure(figsize=dim)
    plt.title(title)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if save_mode:
        plt.savefig("tmp/" + title)
        plt.close()
    else:
        plt.show()
        plt.close()


def learning_mnist(swagans, x, x_cv, number_of_gans, epochs,
             switches=2, print_mode=False,smooth_zero=.1, smooth_one=.9,
             eval_size=1000):

    while number_of_gans > 1:
        d_losses, g_losses = list(), list()
        generators, discriminators = list(), list()
        j=0
        for swagan in swagans:
            d_loss, g_loss = swagan.train(x_train=x,
                                          epochs=epochs,
                                          print_recap=False,
                                          smooth_zero=smooth_zero,
                                          smooth_one=smooth_one)
            d_losses.append(np.mean(d_loss))
            g_losses.append(np.mean(g_loss))
            generators.append(swagan.generator)
            discriminators.append(swagan.discriminator)
            if print_mode:
                images = swagan.generate(100)
                plot_images(images.reshape(100, 28, 28), title=str(number_of_gans) + str(j) + "GAN.png", save_mode=True)
                j += 1

        for i in range(number_of_gans):
            d_l, g_l = swagans[i].evaluate(x=x_cv, batch_size=eval_size)
            d_losses[i] += float(d_l) / (number_of_gans + 1)
            g_losses[i] = g_losses[i] + g_l / (number_of_gans + 1)

        for _ in range(switches):
            swagans, sigma = switching_swagans(swagans)
            d_losses = [d_losses[sigma[i]] for i in range(number_of_gans)]
            g_losses = [g_losses[sigma[i]]/ (number_of_gans+1) for i in range(number_of_gans)]
            discriminators = [discriminators[sigma[i]] for i in range(number_of_gans)]
            for i in range(number_of_gans):
                d_l, g_l = swagans[i].evaluate(x=x_cv, batch_size=eval_size)
                d_losses[i] += float(d_l) / (number_of_gans+1)
                g_losses[i] += float(g_l) / (number_of_gans+1)
        d_to_delete = np.argmax(d_losses)
        g_to_delete = np.argmax(g_losses)

        del generators[g_to_delete]
        del discriminators[d_to_delete]
        print("\n"*2)
        print("best generator loss is " + str(min(g_losses)))
        print("\n"*2)
        for d in d_losses:
            print("f1 score is " + str(d))
        del swagans[g_to_delete]
        number_of_gans += -1
    return swagans
