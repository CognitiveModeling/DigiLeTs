__author__ = "Fedor Scholz"

import os
import sys

from data import read_data, write_data


def extrapolate_participant(participant):
    """extrapolate pressure values for points where this information is missing if possible"""
    for s, symbol in enumerate(participant["trajectories"]):
        for i, instance in enumerate(symbol):
            for p, point in enumerate(instance):
                if point[2] == 0 and point[5] == 0:
                    if len(instance) > p + 2\
                            and instance[p+1][5] == 0\
                            and instance[p+2][5] == 0\
                            and instance[p+1][3] == 0\
                            and instance[p+2][3] == 0:
                        point[2] = 2 * instance[p+1][2]\
                                - instance[p+2][2]
                        # mark that point as extrapolated
                        point[6] = 1
    return participant


if __name__ == '__main__':
    filenames = sys.argv[1:]
    for filename in filenames:
        participant = read_data(filename)
        participant = extrapolate_participant(participant)
        pre, ext = os.path.splitext(filename)
        filename = pre + "_extrapolated" + ext
        write_data(participant, filename)
