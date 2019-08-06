from loadingCGAN.cgan import switching_gans
import numpy as np
from evaluation.evaluation import evaluate


def learning(cgans, x, y, x_cv, y_cv, number_of_gans, epochs,
             switches=2, print_mode=False, mode_d_loss=False,
             reload_images_p=.9, show_past_p=.9, smooth_zero=.1, smooth_one=.9,
             eval_size=1000):
    """
    The exact implementation of SWAGAN algorithm
    Presented in the report (link in README)
    We keep the Generators fixed and shuffle Discriminators (easier to implement) :
    - generator of the first item of cgans will always be the same
    - discriminators are assigned according to the random permutation
    """

    while number_of_gans > 1:
        cv_losses, d_losses, g_losses = list(), list(), list()
        generators, discriminators = list(), list()
        for cgan in cgans:
            cv_loss, d_loss, g_loss = cgan.train(x_train=x,
                                                 y_train=y,
                                                 epochs=epochs,
                                                 print_recap=False,
                                                 reload_images_p=reload_images_p,
                                                 show_past_p=show_past_p,
                                                 smooth_zero=smooth_zero,
                                                 smooth_one=smooth_one)
            cv_losses.append(np.mean(cv_loss))
            d_losses.append(np.mean(d_loss))
            g_losses.append(np.mean(g_loss))
            generators.append(cgan.generator)
            discriminators.append(cgan.discriminator)
        for i in range(number_of_gans):
            d_l, g_l = cgans[i].evaluate(x=x_cv, y=y_cv, batch_size=eval_size, mode_d_loss=mode_d_loss)
            d_losses[i] += float(d_l) / (number_of_gans + 1)
            g_losses[i] = g_losses[i] + g_l / (number_of_gans + 1)
        for _ in range(switches):
            cgans, sigma = switching_gans(cgans)
            cv_losses = [cv_losses[sigma[i]] for i in range(number_of_gans)]
            d_losses = [d_losses[sigma[i]] for i in range(number_of_gans)]
            g_losses = [g_losses[sigma[i]]/ (number_of_gans+1) for i in range(number_of_gans)]
            discriminators = [discriminators[sigma[i]] for i in range(number_of_gans)]
            for i in range(number_of_gans):
                d_l, g_l = cgans[i].evaluate(x=x_cv, y=y_cv, batch_size=eval_size, mode_d_loss=mode_d_loss)
                d_losses[i] += float(d_l) / (number_of_gans+1)
                g_losses[i] = g_losses[i] + g_l / (number_of_gans+1)
        d_to_delete = np.argmax(d_losses)
        g_to_delete = np.argmax(g_losses)
        result_cgan = evaluate(y_true=y_cv, y_pred=cgans[g_to_delete].predict(x=x_cv))
        print("Results of the deleted generator : ")
        print(result_cgan)
        del generators[g_to_delete]
        del discriminators[d_to_delete]
        print("\n"*2)
        print("best generator loss is " + str(min(g_losses)))
        print("\n"*2)
        for d in d_losses:
            print("f1 score is " + str(d))
        if print_mode:
            cgans[g_to_delete].plot_learning()
        del cgans[g_to_delete]
        number_of_gans += -1

        for cgan in cgans:
            result_cgan = evaluate(y_true=y_cv, y_pred=cgan.predict(x=x_cv))
            print(result_cgan)
    return cgans
