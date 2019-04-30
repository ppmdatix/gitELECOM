from matplotlib import pyplot as plt



def save_loss_evolution(malveillance, GANloss, title="title", save_name="image"):
    plt.plot(malveillance, label="malveillance")
    plt.plot(GANloss, label="GANloss")
    plt.legend()
    plt.savefig("images/" + save_name + ".png")
    return True
