from matplotlib import pyplot as plt


def save_loss_evolution(malveillance, GANloss, title="title", save_name="image"):
    plt.plot([10. * m for m in malveillance], label="10 * malveillance")
    plt.plot(GANloss, label="GANloss")
    plt.legend()
    plt.savefig("images/" + save_name + ".png")
    plt.close()
    return True
