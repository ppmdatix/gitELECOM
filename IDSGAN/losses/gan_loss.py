from keras import backend as K
import tensorflow as tf


def hurting_raw(x):
    return (x[0] + 12*x[8] + x[14] * x[23] * x[23]) / 10000.


def hurting(traffic):
    t = tf.transpose(traffic)
    return tf.math.minimum(1., K.mean(hurting_raw(t)))


def linking_loss(link_mode,power=1,alpha=1,offset=0,mult=1,sqrt=2):
    def link(L, hurt):
        if link_mode == "alpha":
            return L * hurt + (1 - hurt) * (alpha * L + offset)
        elif link_mode == "exp":
            return L * hurt + (1 - hurt) * K.exp(L)
        elif link_mode == "pow":
            return L * hurt + (1 - hurt) * K.pow(x=L, a=power)
        elif link_mode == "sum":
            return L + mult * K.pow(hurt, float(1 / sqrt))
        elif link_mode == "hurting":
            return hurt
    return link


def custom_loss(intermediate_output,
                power=1,
                alpha=1,
                offset=0,
                mult=1,
                sqrt=2,
                loss_base="Goodfellow",
                link_mode="alpha"):

    if loss_base == "Goodfellow":
        def loss_function_base(y):
            return - K.log(y)
    elif loss_base == "Wasserstein":
        def loss_function_base(y):
            return - y
    elif loss_base == "Pearson":
        def loss_function_base(y):
            return K.pow(y-1, 2)

    link_f = linking_loss(link_mode=link_mode,
                          power=power,
                          alpha=alpha,
                          offset=offset,
                          mult=mult,
                          sqrt=sqrt,)

    def lossFunction(y_true, y_pred):
        L = loss_function_base(y_pred)
        hurt = hurting(intermediate_output)
        loss = link_f(L=L, hurt=hurt)
        return loss

    return lossFunction


def custom_loss_discriminator(loss_base="Goodfellow"):

    if loss_base == "Goodfellow":
        def loss_function_base(y, y_true):
            return K.mean(- K.log((1-y_true) + (2*y_true - 1)*y))
    elif loss_base == "Wasserstein":
        def loss_function_base(y, y_true):
            return K.mean(- y * (2*y_true - 1))
    elif loss_base == "Pearson":
        def loss_function_base(y, y_true):
            return K.mean((1 - 2*y_true) * K.pow(y-y_true, 2))

    def lossFunction(y_true,y_pred):
        L = loss_function_base(y_pred, y_true)
        return L

    return lossFunction
