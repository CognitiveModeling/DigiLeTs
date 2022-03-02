__author__ = "Julius Wührer, Fedor Scholz"

import os
import pickle
import glob

import numpy as np
import torch
from torchvision.transforms import transforms

from dataset.base_data import BaseDataset
from util import openslice

name = "TrajectoriesDataset"
provides = (("xdiff", "ydiff", "pressure", "stroke_end"), "character", "participant")

# global raw data for potential reusal
data = []



def load_data(path):
    """
    Loads dataset from a folder of individual pickle files for each subject
    :param path: Path to the dataset folder
    """
    global data
    paths = glob.glob(os.path.join(
        path,
        "*.pickle"))
    data = [pickle.load(open(path, "rb")) for path in paths]


def load_transformed_data(path):
    """
    Loads dataset from a single pickle file, generated with data_transformer.py
    :param path: Path to the dataset pickle file
    """
    global data
    data = pickle.load(open(path, "rb"))


class TrajectoriesDataset(BaseDataset):
    """
    Dataset class for the dataset proposed in our paper
    """

    def __init__(self, path, input_size=None, participant_size=None, participant_bounds=openslice(),
                 character_bounds=openslice(), instance_bounds=openslice(), include_participant=True,
                 transform=None, keep=False, load_pretransformed=False):
        """

        :param path: Path to dataset folder or file
        :param input_size: The maximum label for the input, used to generate one-hot encoded labels.
            Can be automatically read from the dataset if left as None
        :param participant_size: The maximum label for the participant, used to generate one-hot encoded labels.
            Can be automatically read from the dataset if left as None
        :param participant_bounds: Which participants samples should be loaded
            (list, slice, set, MultiSlice or other object which implements __contains__)
        :param character_bounds: Which characters samples per participant should be loaded
            (list, slice, set, MultiSlice or other object which implements __contains__)
        :param instance_bounds: Which instances per character and participant should be loaded
            (list, slice, set, MultiSlice or other object which implements __contains__)
        :param include_participant: Whether the output should include a one-hot encoded participant label
        :param transform: The transform to be used when returning samples
        :param keep: Whether the raw data should be kept in memory or not
        :param load_pretransformed: Whether data should be loaded from the dataset folder or pretransformed from a file
        """
        self.transform = transform

        if load_pretransformed:
            # loading pretransformed data from a single file
            if len(data) == 0:
                # only load raw data if we don't already have it loaded
                load_transformed_data(path)
            self.data = data

            trajectories = []
            p_max = 0
            s_max = 0
            for (trajectory, s, p, i) in self.data:
                p_max = max(p_max, p)
                s_max = max(s_max, s)
                if p in participant_bounds and s in character_bounds and i in instance_bounds:
                    trajectories.append((trajectory, s, p, i))
            self.trajectories = trajectories
            if participant_size is None:
                self.participant_size = p_max + 1
            if input_size is None:
                self.input_size = s_max + 1
        else:
            if len(data) == 0:
                # only load raw data if we don't already have it loaded
                load_data(path)

            self.data = data

            if participant_size is None:
                self.participant_size = len(self.data)
            if input_size is None:
                self.input_size = max(map(lambda pdata: len(pdata["trajectories"]), self.data))

            trajectories = []
            for p, participant in enumerate(self.data):
                # only includes participants which were specified in the config
                # or all of them, if config is empty
                if p in participant_bounds:
                    for s, symbol in enumerate(participant["trajectories"]):
                        # only includes characters which were specified in the config
                        # or all of them, i config is empty
                        if s in character_bounds:
                            for i, instance in enumerate(symbol):
                                if i in instance_bounds:
                                    trajectories.append((instance, s, p, i))
            self.trajectories = trajectories

        if input_size is not None:
            self.input_size = input_size
        if participant_size is not None:
            self.participant_size = participant_size

        self.include_participant = include_participant

        # calculating which participant, character, instance is stored at which place in flattened list of samples
        self.pci = np.zeros(self.shape(), np.int)
        for index, (t, c, p, i) in enumerate(self.trajectories):
            self.pci[p][c][i] = index

        if not keep:
            self.data = []

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.trajectories[index])
        else:
            return self.trajectories[index]

    def get_pci(self, p, c, i):
        """
        Gets the sample from the specified participant, character and instance
        :param p: Participant index
        :param c: Character index
        :param i: Instance index
        :return: Sample
        """
        if self.transform:
            return self.transform(self.trajectories[self.pci[p][c][i]])
        else:
            return self.trajectories[self.pci[p][c][i]]

    def collate_trajectories(self, data):
        """
        Collates samples together into batches for the dataloader. Also converts labels to one-hot.
        :param data:
        :return:
        """
        trajectories = [x[0] for x in data]
        targets = torch.nn.utils.rnn.pad_sequence(trajectories)
        characters = [x[1] for x in data]
        characters_onehot = torch.zeros((targets.shape[0], targets.shape[1], self.input_size))
        for i in range(len(characters)):
            characters_onehot[:, i, characters[i]] = 1.0
        participants = [x[2] for x in data]
        participants_onehot = torch.zeros((targets.shape[0], targets.shape[1], self.participant_size))
        if self.include_participant:
            for i in range(len(participants)):
                participants_onehot[:, i, participants[i]] = 1.0
        instances = [x[3] for x in data]
        lengths = torch.LongTensor([len(x) for x in trajectories])
        return targets, characters_onehot, participants_onehot, lengths, (participants, characters, instances)

    @staticmethod
    def default_transform(bridge_size=10):
        return transforms.Compose([Clean(), Diff(), Normalize(), ToTensor()])

    def shape(self):
        return self.participant_size, self.input_size, 5


class ToTensor(object):
    """
    Transform to convert sample trajectory to tensor
    """
    def __call__(self, sample):
        return torch.Tensor(sample[0]), sample[1], sample[2], sample[3]


class ToOnehot(object):
    """
    Transform to convert labels to one-hot encoded vectors.
    Only use if you intend to use the dataset without the dataloader, creating one-hots is taken care of by the collate
    """
    def __init__(self, input_size, participant_size):
        self.input_size = input_size
        self.participant_size = participant_size

    def __call__(self, sample):
        trajectory, character, participant, instance = sample
        trajectory = torch.nn.utils.rnn.pad_sequence([trajectory])
        characters_onehot = torch.zeros((trajectory.shape[0], 1, self.input_size))
        characters_onehot[:, 0, character] = 1.0
        participants_onehot = torch.zeros((trajectory.shape[0], 1, self.participant_size))
        participants_onehot[:, 0, participant] = 1.0
        lengths = torch.LongTensor([len(trajectory)])
        return trajectory, characters_onehot, participants_onehot, lengths, (participant, character, instance)


class Clean(object):
    """
    Transform to clean the dataset, removing trailing nan entries and meta features
    """
    def __call__(self, sample):
        t = []
        last_point_index = np.where(np.isnan(sample[0]))[0][0]
        for i in range(last_point_index - 1):
            if sample[0][i][5] == 0:
                t.append(sample[0][i][:5])
        return np.array(t), sample[1], sample[2], sample[3]


class Diff(object):
    """
    Transform to create an array of difference vectors from an array of positions
    """
    def __call__(self, sample):
        trajectory = sample[0]
        x, y, p, s = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
        x_diff, y_diff = np.hstack(([0], np.diff(x))), np.hstack(([0], np.diff(y)))
        trajectory_diff = np.column_stack((x_diff, y_diff, p, s))
        return trajectory_diff, sample[1], sample[2], sample[3]


class Normalize(object):
    """
    Transform to normalize x and y values to be between 0 and 1, respecting aspect ratio.
    """
    def __call__(self, sample):
        abs_max = np.max(np.abs(sample[0]), axis=0)
        # Don't normalize x and y seperately! This draws out the characters in whatever axis had smaller variation!
        space_max = max(abs_max[0], abs_max[1])
        abs_max[0] = space_max
        abs_max[1] = space_max
        normed = sample[0] / abs_max
        return normed, sample[1], sample[2], sample[3]


def num_to_char(num):
    """
    Converts a character label number to an actual har for readability
    :param num: Character label
    :return: Char belonging to that label
    """
    # characters from 0 to 9 are '0' - '9'
    if num < 10:
        return chr(num + 48)
    # characters from 10 to 35 are 'a' - 'z'
    if num < 36:
        return chr(num + 87)
    # characters from 36 to 61 are 'A' - 'Z'
    if num < 62:
        return chr(num + 29)


def char_to_num(char):
    """
    Converts a char to the corresponding character label number
    :param char: Char to convert
    :return: Character label number
    """
    num = ord(char)
    # characters from '0' to '9' are 0 - 9
    if 47 < num < 58:
        return num - 48
    # characters from 'A' to 'Z' are 36 - 61
    if 64 < num < 90:
        return num - 29
    # characters from 'a' to 'z' are 10 - 35
    if 96 < num < 123:
        return num - 87
