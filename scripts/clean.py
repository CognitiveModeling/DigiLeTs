__author__ = "Fedor Scholz"

import os
import sys

from data import read_data, write_data


def clean_participant(participant):
    # mark all points with pressure = 0 and pen_down = False as cleaned
    for s, symbol in enumerate(participant["trajectories"]):
        for i, instance in enumerate(symbol):
            for p, point in enumerate(instance):
                if point[2] == 0 and point[3] == 0:
                    point[5] = 1

            # adjust timestamps if points were marked as cleaned at the beginning of a trajectory
            for point in instance:
                if point[5] == 0:
                    offset = point[4]
                    break
            for point in instance:
                point[4] -= offset
    return participant


if __name__ == '__main__':
    filenames = sys.argv[1:]
    for filename in filenames:
        participant = read_data(filename)
        participant = clean_participant(participant)
        pre, ext = os.path.splitext(filename)
        filename = pre + "_cleaned" + ext
        write_data(participant, filename)
