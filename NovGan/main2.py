import os
from load_gan import load_gan
from load_cases import load_cases, load_save_name

os.environ["KERAS_BACKEND"] = "tensorflow"

# x_train, x_test = load_data()
epochs = 100
cases = load_cases()

for case in cases:
    print(load_save_name(case))
    generator, discriminator, GAN = load_gan(offset=case["offset"],
                                             alpha=case["alpha"],
                                             randomDim=50,
                                             link_mode=case["link_mode"],
                                             power=case["power"],
                                             mult=case["mult"],
                                             sqrt=case["sqrt"],
                                             loss_base=case["loss_base"])
    """
    _, malveillance, GANloss = trainGAN(discriminator, generator, GAN, X_train=x_train,
                                        epochs=epochs,
                                        batchSize=128,
                                        dLossLimit=0.1,
                                        randomDim=randomDim,
                                        save_mode=True,
                                        save_title=loss_base+"alpha" + str(alpha) + "and_offset" + str(offset))

    evaluation = evaluation(generator=generator,
                            discriminator=discriminator,
                            GAN=GAN)
    
    """