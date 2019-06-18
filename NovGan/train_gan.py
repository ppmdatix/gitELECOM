import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def generateImages(generator,
                   randomDim,
                   examples,
                   reshape=True):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generated_images = generator.predict(noise)
    if reshape:
        generated_images = generated_images.reshape(examples, 28, 28)
    return generated_images


def hurt(image):
    return np.mean(np.mean(((image + 1)/2)))


def train_gan(disc, gen, gan, x_train,
              epochs=1,
              batch_size=128,
              d_loss_limit=0.1,
              randomDim=50,
              reload_images_p=.8,
              show_past_p=.9):
    dL, gL, hing = [], [], []
    to_be_trusted = True
    batch_count = int(x_train.shape[0] / batch_size)
    past_images = generateImages(generator=gen, examples=batch_size, randomDim=randomDim, reshape=False)

    for e in tqdm(range(1, epochs+1)):
        # for e in range(1, epochs+1):
        dloss, gloss, hurting = 0., 0., 0.
        for _ in (range(batch_count)):
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)].reshape((batch_size, 28 * 28))

            noise = np.random.normal(0, 1, size=[batch_size, randomDim])
            if np.random.random() > reload_images_p:
                past_images = generateImages(generator=gen, examples=batch_size, randomDim=randomDim, reshape=False)
            if np.random.random() > show_past_p:
                generated_images = past_images
            else:
                generated_images = gen.predict(noise)
            hurting += np.mean([hurt(gi) for gi in generated_images])

            X = np.concatenate([image_batch, generated_images])

            error = False
            if error:
                X = np.concatenate([generated_images, image_batch])

            y_for_discriminator = np.zeros(2 * batch_size)
            y_for_discriminator[:batch_size] = 0.95
            disc.trainable = True
            dloss += disc.train_on_batch(X, y_for_discriminator)
            disc.trainable = False
            y_for_generator = np.ones(batch_size)
            gloss += gan.train_on_batch(noise, y_for_generator)
        """
        if (dloss/ batch_count) < d_loss_limit and e > epochs/50:
            to_be_trusted = False
            break
        """

        dL.append(dloss / batch_count)
        gL.append(gloss / batch_count)
        hing.append(hurting / batch_count)

    if to_be_trusted:
        return True, dL, gL, hing
    else:
        print("====================ERROR====================")
        print("   The Disc-Loss goesssssssssss 0 my friend")
        print("=============================================")
        return False, [], [], []

