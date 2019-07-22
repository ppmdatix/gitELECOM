from loadingCGAN.swagan_trafic_input import switching_swagans_trafic_input
import numpy as np
from copy import deepcopy as deepcopy



def learning_swagan_trafic_input(swagans, x, x_bad, x_cv, x_bad_cv,
                   number_of_gans, epochs,
                   switches=2, print_mode=False,
                   smooth_zero=.1, smooth_one=.9,
                   eval_size=1000, title="all_data"):

    swagan_base = swagans[0]
    d_loss_base, g_loss_base = swagan_base.train(x_train=x,
                                                 x_train_bad=x_bad,
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
                                          x_train_bad=x_bad,

                                          epochs=epochs,
                                          print_recap=False,
                                          smooth_zero=smooth_zero,
                                          smooth_one=smooth_one)
            d_losses.append(np.mean(d_loss))
            g_losses.append(np.mean(g_loss))
            generators.append(swagan.generator)
            discriminators.append(swagan.discriminator)

        for i in range(number_of_gans):
            d_l, g_l = swagans[i].evaluate(x=x_cv, x_bad=x_bad_cv, batch_size=eval_size)
            d_losses[i] += float(d_l) / (number_of_gans + 1)
            g_losses[i] = g_losses[i] + g_l / (number_of_gans + 1)

        for _ in range(switches):
            swagans, sigma = switching_swagans_trafic_input(swagans)
            d_losses = [d_losses[sigma[i]] for i in range(number_of_gans)]
            g_losses = [g_losses[sigma[i]]/ (number_of_gans+1) for i in range(number_of_gans)]
            discriminators = [discriminators[sigma[i]] for i in range(number_of_gans)]
            for i in range(number_of_gans):
                d_l, g_l = swagans[i].evaluate(x=x_cv, x_bad=x_bad_cv, batch_size=eval_size)
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
