import numpy as np
from matplotlib import pyplot as plt


def zero_or_one(x):
    if x < .5:
        return 0
    else:
        return 1


def false_or_true(x):
    if x[0] < .5 and x[1] < .5:
        return 0
    elif x[0] > x[1]:
        return 0
    elif x[0] == x[1]:
        print("FUCK")
        return 1
    else:
        return 1


def proba_choice(x):
    if x[0] > x[1]:
        return 1 - x[0]
    elif x[0] == x[1]:
        print("FUCK")
        return x[0]
    else:
        return x[1]


def past_labeling(traffics, lab):
    output = []
    size = len(lab)
    for i in range(size):
        label = lab[i]
        output.append(traffics[str(int(label))][i])
    return np.array(output)


def tanh_to_zero_one(x):
    return (1. + x) * .5


def smoothing_y(y_to_smooth, smooth_one, smooth_zero):
    output = list()
    for y in y_to_smooth:
        alpha = np.random.random()
        output.append(y*(smooth_one * alpha + (1-alpha)) + (1-y)*(smooth_zero*alpha))
    return np.array(output).reshape((len(output), 1))


def plot_images(generatedImages, dim=(10,10), title="title"):
    plt.figure(figsize=dim)
    plt.title(title)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()


def creating_dico_index(colnames):
    features = ["num_access_files", "num_shells", "src_bytes",
                "dst_bytes", "root_shell", "num_root", "su_attempted",
                "num_file_creations"]
    output = dict()
    length = len(colnames)
    for feature in features:
        output[feature] = [i for i in range(length) if colnames[i] == feature][0]
    return output


def save_time(duration, location, title):
    with open(location + title + "duration.txt", "w") as text_file:
        text_file.write("Duration: %s" % str(duration))
    return True
