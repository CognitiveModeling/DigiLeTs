__author__ = "Julius Wührer"

"""
Script to pretransform all individual datapoints in our dataset
and save them in a single file to speed up training.
"""
import pickle
import sys

from dataset.trajectories_data import TrajectoriesDataset

if __name__ == '__main__':
    transform = TrajectoriesDataset.default_transform()
    dataset = TrajectoriesDataset("../data/preprocessed/complete/", input_size=62, participant_size=77, transform=transform)
    trajectories = []
    for trajectory in dataset:
        trajectories.append(trajectory)
    pickle.dump(trajectories, open("./data.pickle", "wb"))
    sys.exit()
