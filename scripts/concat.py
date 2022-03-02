__author__ = "Fedor Scholz"

import sys

import pickle
from data import read_data


if __name__ == '__main__':
    dataset = []
    for filename in sys.argv[1:]:
        participant = read_data(filename)
        dataset.append(participant)
    pickle.dump(dataset, open("data/data.pickle", "wb"))
