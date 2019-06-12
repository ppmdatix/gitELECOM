from keras.constraints import Constraint
from keras import backend as K


class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(x=p, min_value=-self.c, max_value=self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

