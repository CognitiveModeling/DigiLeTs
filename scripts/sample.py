__author__ = "Fedor Scholz"

import os
import sys
import numpy as np

from data import read_data, write_data


def get_distribution(participant):
    pressures = []
    for s, symbol in enumerate(participant["trajectories"]):
        for i, instance in enumerate(symbol):
            for p, point in enumerate(instance):
                if point[3] == 1 and point[2] != 0 and point[5] == 0:
                    pressures.append(point[2])
    mean = np.mean(pressures)
    std = np.std(pressures)
    return mean, std


def sample_participant(participant):
    """sample pressure values for points where this information is missing and cannot be extrapolated"""
    dist = get_distribution(participant)
    for s, symbol in enumerate(participant["trajectories"]):
        for i, instance in enumerate(symbol):
            for p, point in enumerate(instance):
                if point[2] == 0 and point[5] == 0:
                    point[2] = np.random.normal(*dist)
                    while point[2] <= 0 or abs(
                            (point[2] - dist[0])/dist[1]) > 1.959963984540:
                        point[2] = np.random.normal(*dist)
                    # note that point is sampled
                    point[7] = 1
    return participant


if __name__ == '__main__':
    filenames = sys.argv[1:]
    for filename in filenames:
        participant = read_data(filename)
        participant = sample_participant(participant)
        pre, ext = os.path.splitext(filename)
        filename = pre + "_sampled" + ext
        write_data(participant, filename)
