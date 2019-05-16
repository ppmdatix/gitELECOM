from loading.loadIDS import loadIDS, trainIDS
from loading.loadGAN import load_gan, load_gan_kdd
from loading.loadingKDD import loadingKDD
from generation.generation import generation_fake_data
from training.training_gan import train_gan, train_gan_kdd
from matplotlib import pyplot as plt

# Parameters
random_dim = 20

# DATA
# x_train, y_train, x_test, y_test = loadData(nrows=100000, attacks=True)
X, Y = loadingKDD(nrows=50000, attack_mode=None, attack=None)
x_test, y_test = X[:3000], Y[:3000]
x_train, y_train = X[:-3000], Y[:-3000]
zero_index_train = [i for y, i in zip(y_train, range(len(y_train))) if y == 0]
zero_index_test = [i for y, i in zip(y_test, range(len(y_test))) if y == 0]

"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5)/127.5
x_test = (x_test.astype(np.float32) - 127.5)/127.5
x_train = x_train.reshape(60000, 784)
"""


data_dim = x_train.shape[1]

# IDS
ids = loadIDS(mode="RandomForest", n_estimators=200,max_depth=15)
trainIDS(ids, x_train=x_train, y_train=y_train)

# GAN
epochs = 1
random_dim = 32
generator, discriminator, gan = load_gan_kdd(data_dim=data_dim, random_dim=random_dim)
# Training GAN
gan_to_be_used, discriminator_loss, generator_loss = train_gan_kdd(disc=discriminator,
                                                                   gen=generator,
                                                                   GAN=gan,
                                                                   random_dim=random_dim,
                                                                   epochs=epochs,
                                                                   x_train=x_train[zero_index_train])




# Testing IDS
number = 100
fake_data = generation_fake_data(generator=generator, number=number, random_dim=random_dim)
real_data = x_test[zero_index_test][:100]
prediction = ids.predict_proba(fake_data)
prediction = [p[0] for p in prediction]
prediction_real = ids.predict_proba(real_data)
prediction_real = [p[0] for p in prediction_real]


plt.hist(prediction, normed=True, label="fake")
plt.hist(prediction_real, label="real", normed=True)
plt.legend()
plt.show()
plt.close()

plt.plot(discriminator_loss, label="disc")
plt.plot(generator_loss, label="gen")
plt.legend()
plt.show()
plt.close()
