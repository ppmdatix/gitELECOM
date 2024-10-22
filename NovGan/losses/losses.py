from keras import backend as K


def hurting(image):
    """
    hurting function that maps the one presented in the report for MNIST images
    """
    return K.mean(K.abs((- image + 1)/2))


def linking_loss(link_mode, power=1, alpha=1, offset=0, mult=1, sqrt=2):
    """
    Testing different way to link hurting and Discriminator fooling
    - in the report we present linear linkage
    """
    def link(l, l_bis):
        if link_mode == "alpha":
            return l * l_bis + (1 - l_bis) * (alpha * l + offset)
        elif link_mode == "exp":
            return l * l_bis + (1 - l_bis) * K.exp(l)
        elif link_mode == "pow":
            return l * l_bis + (1 - l_bis) * K.pow(x=l, a=power)
        elif link_mode == "sum":
            return l + mult * K.pow(l_bis, float(1 / sqrt))
    return link


def custom_loss(intermediate_output,
                power=1,
                alpha=1,
                offset=0,
                mult=1,
                sqrt=2,
                loss_base="Goodfellow",
                link_mode="alpha"):
    """
    Different loss functions tried as explained in the report
    We got the best results with the classical one (binary cross-entropy, named Goodfellow here)
    """

    if loss_base == "Goodfellow":
        def loss_function_base(y):
            return - K.log(K.maximum((y+1)*.5, 1e-9))
    elif loss_base == "Wasserstein":
        def loss_function_base(y):
            return 1 - (y+1)*.5
    elif loss_base == "Pearson":
        def loss_function_base(y):
            return K.pow((y+1)*.5-1, 2)

    link_f = linking_loss(link_mode=link_mode,
                          power=power,
                          alpha=alpha,
                          offset=offset,
                          mult=mult,
                          sqrt=sqrt,)

    def lossFunction(y_true, y_pred):
        """
        Keras compatible loss function
        """
        l = loss_function_base(y_pred)
        l_bis = hurting(intermediate_output)
        loss = link_f(l=l, l_bis=l_bis)
        return loss

    return lossFunction


def custom_loss_discriminator(loss_base="Goodfellow"):
    """
    According to Novgan, discriminator loss does not change
    We slightly change it to prevent mode collapse
    """
    if loss_base == "Goodfellow":
        def loss_function_base(y_true, y_pred):
            return -y_true*K.log(K.maximum((y_pred+1)*.5, 1e-9)) - (1-y_true)*K.log(K.maximum(1-(y_pred+1)*.5, 1e-9))
    elif loss_base == "Wasserstein":
        def loss_function_base(y_true, y_pred):
            return (1 - (2 * y_true - 1) * y_pred) * .5
    elif loss_base == "Pearson":
        def loss_function_base(y_true, y_pred):
            return - y_true * K.pow(y_pred - 1, 2) + (1-y_true) * K.pow(y_pred, 2)

    def lossFunction(y_true, y_pred):
        L = loss_function_base(y_true, y_pred)
        return L

    return lossFunction
