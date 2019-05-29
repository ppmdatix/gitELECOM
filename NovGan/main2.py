import os
from load_gan import load_gan
from load_cases import load_cases, load_save_name
from train_gan import train_gan
from evaluation import evaluation

os.environ["KERAS_BACKEND"] = "tensorflow"

# x_train, x_test = load_data()
epochs = 100
randomDim = 50
batchSize = 128
cases = load_cases()

for case in cases:
    print(load_save_name(case))
    generator, discriminator, GAN = load_gan(offset=case["offset"],
                                             alpha=case["alpha"],
                                             randomDim=randomDim,
                                             link_mode=case["link_mode"],
                                             power=case["power"],
                                             mult=case["mult"],
                                             sqrt=case["sqrt"],
                                             loss_base=case["loss_base"])

    to_be_trusted, disc_loss, gen_loss, hurting = train_gan(disc=discriminator,
                                                            gen=generator,
                                                            x_train=x_train,
                                                            epochs=epochs,
                                                            batchSize=batchSize,
                                                            dLossLimit=0.1,
                                                            randomDim=randomDim)
    if to_be_trusted:
        1
    """
    evaluation = evaluation(generator=generator,
                            discriminator=discriminator,
                            GAN=GAN)
    
    """