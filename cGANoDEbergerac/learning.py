from loadingCGAN.cgan import Cgan, switching_gans
import numpy as np
from keras.losses import binary_crossentropy


def learning(cgans, x, y, x_cv, y_cv, number_of_gans, epochs, switches=2, print_mode=False, mode_d_loss=False):

    while number_of_gans > 1:
        cv_losses, d_losses, g_losses = list(), list(), list()
        generators, discriminators = list(), list()
        for cgan in cgans:
            cv_loss, d_loss, g_loss = cgan.train(x_train=x,
                                                 y_train=y,
                                                 epochs=epochs,
                                                 print_recap=False,
                                                 reload_images_p=.99,
                                                 show_past_p=.95)
            cv_losses.append(np.mean(cv_loss))
            d_losses.append(np.mean(d_loss))
            g_losses.append(np.mean(g_loss))
            generators.append(cgan.generator)
            discriminators.append(cgan.discriminator)
        for _ in range(switches):
            cgans, sigma = switching_gans(cgans)
            cv_losses = [cv_losses[sigma[i]] for i in range(number_of_gans)]
            d_losses = [d_losses[sigma[i]] for i in range(number_of_gans)]
            g_losses = [g_losses[sigma[i]] for i in range(number_of_gans)]
            discriminators = [discriminators[sigma[i]] for i in range(number_of_gans)]
            for i in range(number_of_gans):
                d_l, g_l = cgans[i].evaluate(x=x_cv, y=y_cv, batch_size=100, mode_d_loss=mode_d_loss)
                d_losses[i] += float(d_l),
                g_losses[i] = g_losses[i] + g_l
        d_to_delete = np.argmax(d_losses)
        g_to_delete = np.argmax(g_losses)
        del generators[g_to_delete]
        del discriminators[d_to_delete]
        if print_mode:
            cgans[g_to_delete].plot_learning()
        del cgans[g_to_delete]
        number_of_gans += -1
        cgans, _ = switching_gans(cgans)
    return cgans
