__author__ = "Fedor Scholz"

import os
import sys
import pickle

import numpy as np


def line_to_instance(line):
    """Convert a string in original data format into an instance array"""
    line = line.split(" ")
    instance = np.empty((250, 8))
    instance[:] = np.nan
    # iterate over points
    length = 0
    for p in range(len(line)//5):
        # iterate over data: x, y, pressure, pen_down, time
        for d in range(5):
            instance[p, d] = float(line[p*5 + d])
        instance[p, 5:] = 0
        length += 1
    return instance, length


def instance_to_line(instance):
    """Convert an instance array back into a string in original data format"""
    line_data = ""
    line_info = ""
    i = 0
    while not np.isnan(instance[i][0]):
        if instance[i][5] == 0:
            line_data += "{:f} ".format(instance[i, 0])
            line_data += "{:f} ".format(instance[i, 1])
            line_data += "{:f} ".format(instance[i, 2])
            line_data += str(int(instance[i, 3]))
            line_data += " "
            line_data += "{:f} ".format(instance[i, 4])
        line_info += str(int(instance[i, 5]))
        line_info += " "
        line_info += str(int(instance[i, 6]))
        line_info += " "
        line_info += str(int(instance[i, 7]))
        line_info += " "
        i += 1
    line_data = line_data[:-1]
    line_data += "\n"
    line_info = line_info[:-1]
    line_info += "\n"
    return line_data, line_info


def lines_to_participant(lines):
    """Convert multiple strings in original data format into a participant array"""
    participant = np.empty((62, 5, 250, 8))
    lengths = np.empty((62, 5), dtype=np.int8)
    # x, y, pressure, down, time, deleted, extrapolated, sampled
    participant[:] = np.nan
    # iterate over symbols
    for s in range(62):
        # iterate over instance
        for i in range(5):
            participant[s, i], lengths[s, i] = line_to_instance(lines[s*5 + i])
    return participant, lengths


def create_label_line(s):
    label = np.zeros([62])
    label[s] = 1
    line = ""
    for l in label:
        line += "{} ".format(l)
    line = line[:-1]
    line += "\n"
    return line


def participant_to_lines(participant):
    """Convert a participant array back into multiple string in original data format"""
    lines_data = []
    lines_info = []
    # iterate over symbols
    for s in range(62):
        # iterate over instance
        for i in range(5):
            line_data, line_info = instance_to_line(participant[s, i])
            lines_data.append(line_data)
            lines_data.append(create_label_line(s))
            lines_info.append(line_info)
    return lines_data, lines_info


def read_original_data(filename):
    print("Reading from: " + str(filename))
    # extract info about participant
    info = os.path.split(filename)[-1].split("_")
    attrs = info[0].split("-")
    wid = attrs[0]
    gender = attrs[1]
    age = attrs[2]
    hand = attrs[3]
    time = info[1]
    # read data
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    trajectories, lengths = lines_to_participant(lines[::2])
    return {"wid": wid, "gender": gender, "age": age, "hand": hand,
            "time": time, "trajectories": trajectories, "lengths": lengths}


def write_original_data(participant, filename):
    print("Writing to: " + filename)
    lines_data, lines_info = participant_to_lines(participant["trajectories"])
    f = open(filename, "w")
    f.writelines(lines_data)
    f.close()
    f = open(filename + "_info", "w")
    f.writelines(lines_info)
    f.close()


def read_data(filename):
    print("Reading from: " + str(filename))
    return pickle.load(open(filename, "rb"))


def write_data(participant, filename):
    print("Writing to: " + filename)
    pickle.dump(participant, open(filename, "wb"))


if __name__ == '__main__':
    filenames = sys.argv[1:]
    for filename in filenames:
        participant = read_original_data(filename)
        write_data(participant, filename + ".pickle")
