import numpy as np


def zero_or_one(x):
    if x < .5:
        return 0
    else:
        return 1


def false_or_true(x):
    if x[0] < .5 and x[1] < .5:
        return 0
    elif x[0] > x[1]:
        return 0
    elif x[0] == x[1]:
        print("FUCK")
        return 1
    else:
        return 1


def proba_choice(x):
    if x[0] > x[1]:
        return 1 - x[0]
    elif x[0] == x[1]:
        print("FUCK")
        return x[0]
    else:
        return x[1]


def past_labeling(traffics, lab):
    output = []
    size = len(lab)
    for i in range(size):
        label = lab[i]
        output.append(traffics[str(int(label))][i])
    return np.array(output)


def tanh_to_zero_one(x):
    return (1. + x) * .5