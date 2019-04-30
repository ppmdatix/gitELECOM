from loading.loadData import loadData
from loading.loadIDS import loadIDS, trainIDS
from loading.loadGAN import load_gan
from generation.generation import generation_fake_data
from training.training_gan import train_gan
from matplotlib import pyplot as plt

# Parameters
random_dim = 20

# DATA
x_train, y_train, x_test, y_test = loadData(nrows=100000, attacks=True)
zero_index_train = [i for y, i in zip(y_train, range(len(y_train))) if y == 0]

print(x_test.shape)
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5)/127.5
x_test = (x_test.astype(np.float32) - 127.5)/127.5
x_train = x_train.reshape(60000, 784)
"""


data_dim = x_train.shape[1]

# IDS
ids = loadIDS(mode="RandomForest")
trainIDS(ids, x_train=x_train, y_train=y_train)

# GAN
epochs = 200
generator, discriminator, gan = load_gan(data_dim=data_dim, random_dim=random_dim)
gan_to_be_used, discriminator_loss, generator_loss = train_gan(disc=discriminator, gen=generator, GAN=gan, random_dim=random_dim, epochs=epochs, x_train=x_train[zero_index_train])

# Training GAN


# Testing IDS
number = 100
fake_data = generation_fake_data(generator=generator, number=number, random_dim=random_dim)
prediction = ids.predict(fake_data)


plt.hist(prediction)
plt.show()
plt.close()

plt.plot(discriminator_loss, label="disc")
plt.plot(generator_loss, label="gen")
plt.legend()
plt.show()
plt.close()
