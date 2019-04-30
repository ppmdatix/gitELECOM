from ploting import saveImages
from save_loss_evolution import save_loss_evolution
from save_histogram_malveillance import save_histogram_malveillance


def save_data(dico, title, save_name, generator, X_train, randomDim):
    saveImages(dico["images"], title=title,
               save_name=save_name+"IMAGES")
    save_loss_evolution(malveillance=dico["malveillance"],
                        GANloss=dico["GANloss"],
                        title=title + " LOSS EVOLUTION",
                        save_name=save_name+"LOSS_EVOLUTION")
    save_histogram_malveillance(generator=generator, X_train=X_train, randomDim=randomDim,
                                examples=10000, save_name=save_name+"HISTO")
    return True
