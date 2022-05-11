import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, noise_levels, name='figure',
              model_name=['Baseline', 'Pre-processing', 'In-processing', 'Post-processing'],
              save=False, x_label='x', y_label='y', title='title', x_lim=None):
    """

    :param data:
    :param noise_levels:
    :param name:
    :param model_name:
    :param save:
    :param x_label:
    :param y_label:
    :param title:
    :param x_lim:
    :return: the plotting instance
    """
    num_models = len(model_name)
    num_levels = len(noise_levels)
    colours = {'red': '-r', 'green': '-g', 'blue': '-b', 'black': '-k'}
    means = np.zeros((num_models, num_levels + 1))
    mean_er = np.zeros((num_models, num_levels + 1))
    x = np.insert(noise_levels, 0, 0)
    for i in range(0, num_models):
        for j in range(0, len(means[0])):
            means[i, j] = np.mean(data[i, j])
            mean_er[i, j] = np.std(data[i, j]) / np.sqrt(len(data[i, j]))
    f, ax = plt.subplots()
    for i in range(0, 4):
        ax.plot(x, means[i], list(colours.values())[i], label=model_name[i])
        ax.fill_between(x, means[i] - mean_er[i], means[i] + mean_er[i], alpha=0.25, color=list(colours.keys())[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if x_lim is not None:
        plt.xlim([])

    if save:
        plt.savefig(name)
    else:
        plt.show()

    return ax
