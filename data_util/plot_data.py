import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, noise_levels, name='figure',
              model_name=['Baseline', 'Pre-processing', 'In-processing', 'Post-processing'],
              save=False, x_label='x', y_label='y', title=None, x_lim=None, y_lim=None):
    """
    plots data as a line graph for each model
    :param data: input data (y-axis)
    :param noise_levels: differing levels of noise for each input data (x-axis)
    :param name: defines where the figure should be saved and with what name
    :param model_name: different data set names for the legend
    :param save: if true, save the plot
    :param x_label: x axis label
    :param y_label: y axis label
    :param title: title of the plot
    :param x_lim: list [a, b] containing the limits of the x-axis
    :param y_lim: list [a, b] containing the limits of the y-axis
    :return: plot object
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

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('Difference of {}'.format(y_label))

    if title is not None:
        plt.title(title)

    if x_lim is not None:
        plt.xlim(x_lim)

    if y_lim is not None:
        plt.ylim(y_lim)

    if save:
        plt.savefig(name)
    else:
        plt.show()

    return ax
