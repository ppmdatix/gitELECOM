from loading.loadIDS import loadIDS, trainIDS
from loading.loadGAN import load_gan, load_gan_kdd
from loading.loadingKDD import loadingKDD
from generation.generation import generation_fake_data
from training.training_gan import train_gan, train_gan_kdd
from matplotlib import pyplot as plt
from losses.gan_loss import hurting_raw
import numpy as np

# Parameters

# DATA
# x_train, y_train, x_test, y_test = loadData(nrows=100000, attacks=True)
attack_mode = None
X, Y, colnames = loadingKDD(nrows=100000, attack_mode=attack_mode, attack=None)
# assert len(colnames) == 122, "You did not load enough data"
x_test, y_test = X[:3000], Y[:3000]
x_train, y_train = X[:-3000], Y[:-3000]
zero_index_train = [i for y, i in zip(y_train, range(len(y_train))) if y == 0]
zero_index_test = [i for y, i in zip(y_test, range(len(y_test))) if y == 0]
one_index_train = [i for y, i in zip(y_train, range(len(y_train))) if y == 1]
one_index_test = [i for y, i in zip(y_test, range(len(y_test))) if y == 1]
balanced_size = min(len(zero_index_train), len(one_index_train))
if attack_mode is None:
    x_balanced_train = np.concatenate((x_train[zero_index_train][:balanced_size],
                                       x_train[one_index_train][:balanced_size]))
    y_balanced_train = np.concatenate((y_train[zero_index_train][:balanced_size],
                                       y_train[one_index_train][:balanced_size]))
elif attack_mode is False:
    x_balanced_train = x_train[zero_index_train]
    y_balanced_train = y_train[zero_index_train]
elif attack_mode:
    x_balanced_train = x_train[one_index_train]
    y_balanced_train = y_train[one_index_train]

"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5)/127.5
x_test = (x_test.astype(np.float32) - 127.5)/127.5
x_train = x_train.reshape(60000, 784)
"""


data_dim = x_train.shape[1]

# IDS
ids = loadIDS(mode="RandomForest", n_estimators=200,max_depth=15)
trainIDS(ids, x_train=x_balanced_train, y_train=y_balanced_train)

# GAN
epochs = 20
random_dim = 32
link_mode = "alpha"
loss_base = "Wasserstein"
generator, discriminator, gan = load_gan_kdd(data_dim=data_dim,
                                             random_dim=random_dim,
                                             offset=.0,
                                             alpha=5.,
                                             link_mode="alpha",
                                             loss_base=loss_base)

number = 100
fake_data_first = generation_fake_data(generator=generator, number=number, random_dim=random_dim)
# Training GAN
gan_to_be_used, discriminator_loss, generator_loss = train_gan_kdd(disc=discriminator,
                                                                   dLossLimit=-1000,
                                                                   gen=generator,
                                                                   GAN=gan,
                                                                   random_dim=random_dim,
                                                                   epochs=epochs,
                                                                   x_train=x_train[zero_index_train])


# Testing IDS
number = 100
fake_data = generation_fake_data(generator=generator, number=number, random_dim=random_dim)
real_data = x_test[zero_index_test][:number]
real_data_one = x_test[one_index_test][:number]
prediction = ids.predict_proba(fake_data)
prediction = [p[1] for p in prediction]
prediction_real = ids.predict_proba(real_data)
prediction_real = [p[1] for p in prediction_real]
prediction_real_one = ids.predict_proba(real_data_one)
prediction_real_one = [p[1] for p in prediction_real_one]


plt.hist(prediction, density=True, label="fake")
plt.hist(prediction_real, label="real_0", density=True)
plt.hist(prediction_real_one, label="real_1", density=True)
plt.legend()
plt.show()
plt.close()

plt.plot(discriminator_loss, label="disc")
plt.plot(generator_loss, label="gen")
plt.legend()
plt.show()
plt.close()


plt.hist([min(1,hurting_raw(f)) for f in fake_data], label="hurting", density=True)
plt.hist([min(1,hurting_raw(f)) for f in fake_data_first], label="hurting first", density=True)
plt.hist([min(1,hurting_raw(f)) for f in real_data_one], label="hurting x test 1", density=True)
plt.hist([min(1,hurting_raw(f)) for f in real_data], label="hurting x test 0", density=True)
plt.legend()
plt.show()
plt.close()