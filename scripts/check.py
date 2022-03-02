__author__ = "Fedor Scholz"

import sys
import numpy as np

from data import read_data

scribble = ["003", "006", "009", "016", "023", "047", "059", "063", "097", "101"]


def get_out_of_bounds(dataset):
    res = []
    for participant in dataset:
        for c1, s in enumerate(participant["trajectories"]):
            for c2, i in enumerate(s):
                for c3 in range(participant["lengths"][c1, c2]):
                    if i[c3, 0] < 0 or i[c3, 0] > 1 or i[c3, 1] < 0 or i[c3, 1] > 1:
                        res.append((participant["wid"], c1, c2, c3))
    return res


def get_pressure_0_at_beginning(dataset):
    res = []
    for participant in dataset:
        for c1, s in enumerate(participant["trajectories"]):
            for c2, i in enumerate(s):
                if i[0, 2] == 0:
                    res.append((participant["wid"], c1, c2))
    return res


def get_no_pen_downs(dataset):
    res = []
    for participant in dataset:
        for c1, s in enumerate(participant["trajectories"]):
            for c2, i in enumerate(s):
                pen_downs = 0
                for c3 in range(participant["lengths"][c1, c2]):
                    pen_downs += i[c3, 3]
                if pen_downs == 0:
                    res.append((participant["wid"], c1, c2))
    return res


def get_no_pen_down_at_beginning(dataset):
    res = []
    for participant in dataset:
        for c1, s in enumerate(participant["trajectories"]):
            for c2, i in enumerate(s):
                if i[0, 3] != 1:
                    res.append((participant["wid"], c1, c2))
    return res


def get_timestamp_0_in_the_middle(dataset):
    res = []
    for participant in dataset:
        for c1, s in enumerate(participant["trajectories"]):
            for c2, i in enumerate(s):
                for c3 in range(participant["lengths"][c1, c2]):
                    if i[c3, 4] == 0 and c3 != 0:
                        res.append((participant["wid"], c1, c2, c3))
    return res


def get_timestamp_non_0_at_beginning(dataset):
    res = []
    for participant in dataset:
        for c1, s in enumerate(participant["trajectories"]):
            for c2, i in enumerate(s):
                if i[0, 4] != 0:
                    res.append((participant["wid"], c1, c2))
    return res


def get_shady_data_points(dataset):
    res = []
    for r in get_out_of_bounds(dataset):
        res.append(r)
    for r in get_pressure_0_at_beginning(dataset):
        res.append(r)
    for r in get_no_pen_downs(dataset):
        res.append(r)
    for r in get_no_pen_down_at_beginning(dataset):
        res.append(r)
    for r in get_timestamp_0_in_the_middle(dataset):
        res.append(r)
    for r in get_timestamp_non_0_at_beginning(dataset):
        res.append(r)
    return res


if __name__ == '__main__':
    filenames = sys.argv[1:]
    dataset = []
    for filename in filenames:
        participant = read_data(filename)
        if participant["wid"] not in scribble:
            dataset.append(participant)
    dataset.sort(key=lambda x: x["wid"])
    print()

    out_of_bounds = get_out_of_bounds(dataset)
    if len(out_of_bounds) > 0:
        print("Out of bounds:")
        for r in out_of_bounds:
            print(r)
        print()

    pressure_0_at_beginning = get_pressure_0_at_beginning(dataset)
    if len(pressure_0_at_beginning) > 0:
        print("Pressure 0 at beginning:")
        for r in pressure_0_at_beginning:
            print(r)
        print()

    no_pen_downs = get_no_pen_downs(dataset)
    if len(no_pen_downs) > 0:
        print("No pen downs:")
        for r in no_pen_downs:
            print(r)
        print()

    no_pen_down_at_beginning = get_no_pen_down_at_beginning(dataset)
    if len(no_pen_down_at_beginning) > 0:
        print("No pen downs:")
        for r in no_pen_down_at_beginning:
            print(r)
        print()

    timestamp_0_in_the_middle = get_timestamp_0_in_the_middle(dataset)
    if len(timestamp_0_in_the_middle) > 0:
        print("Timestamp 0 in the middle:")
        for r in timestamp_0_in_the_middle:
            print(r)
        print()

    timestamp_non_0_at_beginning = get_timestamp_non_0_at_beginning(dataset)
    if len(timestamp_non_0_at_beginning) > 0:
        print("Timestamp non 0 at beginning:")
        for r in timestamp_non_0_at_beginning:
            print(r)
        print()
