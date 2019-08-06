from loadingCGAN.swagan import switching_swagans
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as deepcopy


def plot_images(generated_images,
                dim=(10,10),
                title="title",
                save_mode=False):
    """

    :param generated_images: data to plot
    :param dim: size of the displayed pictures
    :param title: - - -
    :param save_mode: - - -
    :return:
    """
    plt.figure(figsize=dim)
    plt.title(title)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if save_mode:
        plt.savefig( title)
        plt.close()
    else:
        plt.show()
        plt.close()
    return True


def learning_mnist(swagans, x, x_cv,
                   number_of_gans, epochs,
                   switches=2, print_mode=False,
                   smooth_zero=.1, smooth_one=.9,
                   eval_size=1000, title="all_data"):
    """
    The exact implementation of SWAGAN algorithm
    Presented in the report (link in README)
    We keep the Generators fixed and shuffle Discriminators (easier to implement) :
    - generator of the first item of cgans will always be the same
    - discriminators are assigned according to the random permutation

    Quite similar to learning.py but adapted to MNIST dataset
    """

    swagan_base = swagans[0]
    d_loss_base, g_loss_base = swagan_base.train(x_train=x,
                                  epochs=epochs*number_of_gans,
                                  print_recap=False,
                                  smooth_zero=smooth_zero,
                                  smooth_one=smooth_one)


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
                # pwd = os.getcwd() + "/"
                pwd = "tmp/" + title + str(number_of_gans) + str(j) + "GAN.png"
                plot_images(images.reshape(100, 28, 28), title=pwd, save_mode=True)
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
    return swagans, swagan_base
