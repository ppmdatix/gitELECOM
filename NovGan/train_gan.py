import numpy as np
from tqdm import tqdm


def generateImages(generator,
                   randomDim,
                   examples):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)
    return generatedImages


def hurt(image):
    return np.mean(np.mean(np.abs((- image + 1)/2)))


def train_gan(disc, gen, gan, x_train,
             epochs=1,
             batchSize=128,
             dLossLimit=0.1,
             randomDim=50):
    dL, gL, hing = [], [], []
    toBeTrusted = True
    batchCount = int(x_train.shape[0] / batchSize)

    for e in tqdm(range(1, epochs+1)):
        dloss, gloss, hurting = 0., 0., 0.
        for _ in (range(batchCount)):
            imageBatch = x_train[np.random.randint(0, x_train.shape[0], size=batchSize)]
            generatedImages = generateImages(generator=gen, randomDim=randomDim, examples=batchSize)
            hurting += np.mean([hurt(gi) for gi in generatedImages])
            X = np.concatenate([imageBatch, generatedImages])
            yDis = np.zeros(2*batchSize)
            yDis[:batchSize] = 0.9
            disc.trainable = True
            dloss += disc.train_on_batch(X, yDis)
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            disc.trainable = False
            gloss += gan.train_on_batch(noise, yGen)

        if dloss < dLossLimit:
            toBeTrusted = False
            break

        dL.append(dloss / batchCount)
        gL.append(gloss / batchCount)
        hing.append(hurting / batchCount)

    if toBeTrusted:
        return True, dL, gL, hing
    else:
        print("====================ERROR====================")
        print("   The Disc-Loss goesssssssssss 0 my friend")
        print("=============================================")
        return False, [], [], []

