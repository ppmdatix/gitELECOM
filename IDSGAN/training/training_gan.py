import numpy as np
from generation.generation import generation_fake_data
from tqdm import tqdm


def train_gan(disc, gen, GAN,
              x_train,
              random_dim,
              epochs=1,
              batchSize=128,
              dLossLimit=0.1):
    dL, gL = [], []

    to_be_trusted = True
    batchCount = int(x_train.shape[0] / batchSize)

    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(int(batchCount))):
            # Get a random set of input noise and images
            index = [np.random.randint(0, x_train.shape[0], size=batchSize)]
            real_data = x_train[index]
            fake_data = generation_fake_data(generator=gen, number=batchSize, random_dim=random_dim)
            X = np.concatenate([real_data, fake_data])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            disc.trainable = True
            dloss = disc.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, random_dim])
            yGen = np.ones(batchSize)
            disc.trainable = False
            gloss = GAN.train_on_batch(noise, yGen)

        if dloss < dLossLimit:
            to_be_trusted = False
            break
        # Store loss of most recent batch from this epoch
        dL.append(dloss)
        gL.append(gloss)

        if e == 1 or e % 20 == 0:
            save_mode = True
            # saveModels(e, generator=gen, discriminator=disc)
            # plotGeneratedImages(e, generator=gen, randomDim=random_dim, save_mode=save_mode)
        else:
            save_mode = False

    # Plot losses from every epoch
    if to_be_trusted:
        # plotLoss(e, dLosses=dL, gLosses=gL)
        return True, dL, gL
    else:
        print("=========ERROR=========")
        print("The Disc-Loss goesssssssssss ZERO friend")
        return False, [], []


def train_gan_kdd(disc,
                  gen,
                  GAN,
                  x_train,
                  random_dim,
                  epochs=1,
                  batchSize=128,
                  dLossLimit=0.1,
                  smooth_zero_min=0.,
                  smooth_one_min=.99,
                  smooth_zero_max=.05,
                  smooth_one_max=1.05,
                  reload_data_p = .8,
                  show_past_p = .99):

    dL, gL = [], []

    to_be_trusted = True
    batchCount = int(x_train.shape[0] / batchSize)

    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    past_images = generation_fake_data(generator=gen, number=batchSize, random_dim=random_dim)
    for _ in tqdm(range(1, epochs+1)):
        for _ in range(int(batchCount)):
            # Train discriminator
            if np.random.random() > reload_data_p:
                past_images = generation_fake_data(generator=gen, number=batchSize, random_dim=random_dim)
            # Get a random set of input noise and images
            index = [np.random.randint(0, x_train.shape[0], size=batchSize)]
            real_data = x_train[index]
            if np.random.random() > show_past_p:
                fake_data = past_images
            else:
                fake_data = generation_fake_data(generator=gen, number=batchSize, random_dim=random_dim)
            X = np.concatenate([real_data, fake_data])

            # Labels for generated and real data, with smoothing
            yDis = np.zeros(2*batchSize)
            for i in range(batchSize):
                alpha = np.random.random()
                yDis[batchSize + i] = smooth_zero_min*alpha + smooth_zero_max*(1-alpha)
                yDis[i] = smooth_one_min*alpha + smooth_one_max*(1-alpha)

            disc.trainable = True
            dloss = disc.train_on_batch(X, yDis)
            disc.trainable = False

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, random_dim])
            yGen = np.ones(batchSize)
            gloss = GAN.train_on_batch(noise, yGen)

        if dloss < dLossLimit:
            to_be_trusted = False
            break
        # Store loss of most recent batch from this epoch
        dL.append(dloss)
        gL.append(gloss)

    # Plot losses from every epoch
    if to_be_trusted:
        # plotLoss(e, dLosses=dL, gLosses=gL)
        return True, dL, gL
    else:
        print("=========ERROR=========")
        print("The Disc-Loss goesssssssssss ZERO my friend")
        return False, [], []
