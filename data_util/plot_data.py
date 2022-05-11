import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, name='data', model_name=['Baseline', 'Pre-processing', 'In-processing', 'Post-processing'],
              save=False, dir='data/'):
    num_models = len(model_name)
    colours = {'red': '-r', 'green': '-g', 'blue': '-b', 'black': '-k'}
    means = np.zeros((4, 21))
    mean_er = np.zeros((4, 21))
    x = np.arange(0, 21)
    for i in range(0, num_models):
        for j in range(0, len(means[0])):
            means[i, j] = np.mean(data[i, j])
            mean_er[i, j] = np.std(data[i, j]) / np.sqrt(len(data[i, j]))

    for i in range(0, 4):
        plt.plot(x, means[i], list(colours.values())[i], label=model_name[i])
        plt.fill_between(x, means[i] - mean_er[i], means[i] + mean_er[i], alpha=0.25, color=list(colours.keys())[i])

    if save:
        plt.savefig(dir + name)
    else:
        plt.show()
