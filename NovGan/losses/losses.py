from keras import backend as K


def hurting(image):
    return K.mean(K.abs((- image + 1)/2))


def linking_loss(link_mode,power=1,alpha=1,offset=0,mult=1,sqrt=2):
    def link(L, L_bis):
        if link_mode == "alpha":
            return L * L_bis + (1-L_bis) * (alpha*L + offset)
        elif link_mode == "exp":
            return L * L_bis + (1-L_bis) * K.exp(L)
        elif link_mode == "pow":
            return L * L_bis + (1-L_bis) * K.pow(x=L, a=power)
        elif link_mode == "sum":
            return L + mult * K.pow(L_bis, float(1/sqrt))
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

    def lossFunction(y_true,y_pred):
        L = loss_function_base(y_pred)
        L_bis = hurting(intermediate_output)
        loss = link_f(L=L, L_bis=L_bis)
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
