from keras import backend as K

def customLoss(layer_weights, lamda, alpha=1, y_func="pow10"):

    def lossFunction(y_true,y_pred):
        y = - K.log( K.abs(y_pred - y_true))
        x = lamda * (K.mean(K.square(( - layer_weights + 1))))#/2)) + 0.5*K.std(layer_weights)) / 2
        if y_func == "carre":
            fy = y*y
        elif y_func == "pow10":
            fy = y**10
        else:
            fy = y
        #loss = x * alpha * y + (1 - x)*fy
        loss = x + y
        return - loss

    return lossFunction