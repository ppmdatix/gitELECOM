import numpy as np


def generation_fake_data(generator, number, random_dim):
    noise = np.random.normal(0, 1, size=[number, random_dim])
    fake_data = generator.predict(noise)

    return fake_data
