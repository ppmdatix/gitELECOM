import numpy as np


def eliminate(cgans, criteria="g_loss", mode="max", window=10):




    assert (len(cgans) > 1), "There is 1 or 0 CGANS left"
    if mode == "max":
        func = np.argmax
    elif mode == "min":
        func = np.argmin
    arg = func([np.mean(cgan.history[criteria][-window:]) for cgan in cgans])
    del cgans[arg]
    return cgans







