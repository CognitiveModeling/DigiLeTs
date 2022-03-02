__author__ = "Fedor Scholz"

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from data import read_data

rect_x1 = 0
rect_y1 = 0
rect_x2 = 1
rect_y2 = 1

factor = 20


def plot_instance(instance, ax):
    ax.set_facecolor("white")

    # plot lines
    for i in range(len(instance) - 1):
        if instance[i][5] == 0 and instance[i+1][5] == 0:
            if instance[i+1][3] == 0:
                color = "black"
            elif instance[i+1][3] == 1:
                color = "yellow"
        else:
            color = "magenta"
        ax.plot([instance[i][0], instance[i+1][0]],
                [instance[i][1], instance[i+1][1]],
                linewidth=instance[i][2]*10,
                solid_capstyle="round",
                color=color,
                alpha=0.5)

    # plot points
    for i in range(len(instance)):
        if instance[i][5] == 1:
            # this point is marked as to be cleaned
            ax.plot(instance[i][0], instance[i][1],
                    marker=".",
                    markersize=10,
                    color="magenta",
                    alpha=1)
        elif instance[i][6] == 0 and instance[i][7] == 0:
            # this is a usual point
            ax.plot(instance[i][0], instance[i][1],
                    marker=".",
                    markersize=10,
                    color="black",
                    alpha=0.25)
        elif instance[i][6] == 1 and instance[i][7] == 0:
            # this point's pressure was extrapolated
            ax.plot(instance[i][0], instance[i][1],
                    marker=".",
                    markersize=10,
                    color="green",
                    alpha=1)
        elif instance[i][6] == 0 and instance[i][7] == 1:
            # this point's pressure was sampled
            ax.plot(instance[i][0], instance[i][1],
                    marker=".",
                    markersize=10,
                    color="blue",
                    alpha=1)

    cleaned = np.nansum(instance[:, 5])
    extrapolated = np.nansum(instance[:, 6])
    sampled = np.nansum(instance[:, 7])

    color = "black"
    if extrapolated != 0:
        color = "green"
    ax.text(
        rect_x1+0.05,
        rect_y1+0.95,
        "extrapolated: {}".format(extrapolated),
        color=color)

    color = "black"
    if sampled != 0:
        color = "blue"
    ax.text(
        rect_x1+0.05,
        rect_y1+0.9,
        "sampled: {}".format(sampled),
        color=color)

    color = "black"
    if cleaned != 0:
        color = "magenta"
    ax.text(
        rect_x1+0.05,
        rect_y1+0.85,
        "cleaned: {}".format(cleaned),
        color=color)

    ax.text(rect_x1+0.05, rect_y1+0.05,
            "#trajectories: {}".format(np.nansum(instance[:, 3])))
    ax.axis([rect_x1, rect_x2, rect_y1, rect_y2])
    return ax


def plot(participant):
    plt.clf()
    _, axs = plt.subplots(nrows=62, ncols=5, figsize=(4*5, 4*62), sharex=True, sharey=True)

    for s, symbol in enumerate(participant["trajectories"]):
        for i, instance in enumerate(symbol):
            axs[s, i] = plot_instance(instance[:participant["lengths"][s, i]], axs[s, i])

    plt.tight_layout()


if __name__ == '__main__':
    filenames = sys.argv[1:]
    for filename in filenames:
        participant = read_data(filename)
        plot(participant)
        filename = os.path.splitext(filename)[0] + ".png"
        print("Writing to: " + filename)
        plt.savefig(filename)
