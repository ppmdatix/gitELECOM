import os
import numpy as np
from load_gan import load_gan
from load_cases import load_cases, load_save_name
from train_gan import train_gan
from evaluation import evaluation
from keras.datasets import mnist

os.environ["KERAS_BACKEND"] = "tensorflow"

(x_train, _), (x_test, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5)/127.5
x_test = (x_test.astype(np.float32) - 127.5)/127.5
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

epochs = 100
randomDim = 32
batchSize = 128
examples = 100
number_swagans = 2
cases = load_cases(loss_bases=["Goodfellow"],alphas=[1.],offsets=[0.])

for case in cases:
    print(case)
    save_name = load_save_name(case)
    swagans = [load_gan(offset=case["offset"],
                                             alpha=case["alpha"],
                                             randomDim=randomDim,
                                             link_mode=case["link_mode"],
                                             power=case["power"],
                                             mult=case["mult"],
                                             sqrt=case["sqrt"],
                                             loss_base=case["loss_base"]) for _ in range(number_swagans)]

    to_be_trusted, disc_loss, gen_loss, hurting = train_gan(disc=discriminator,
                                                            gen=generator,
                                                            gan=gan,
                                                            x_train=x_train,
                                                            epochs=epochs,
                                                            batch_size=batchSize,
                                                            d_loss_limit=0.05,
                                                            randomDim=randomDim)
    if to_be_trusted:
        location = "results/" + case["loss_base"] + "/" + case["link_mode"] + "/" + save_name
        evaluation(generator=generator,
                   randomDim=randomDim,
                   location=location,
                   examples=examples, title=save_name,
                   hurting=hurting, gen_loss=gen_loss, disc_loss=disc_loss, x_test=x_test[:examples*examples])

print("LETS GO TO SLEEP \n "*10)
