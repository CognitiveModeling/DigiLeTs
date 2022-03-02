__author__ = "Fedor Scholz"

import argparse
import sys

from data import read_original_data, write_data, write_original_data
from clean import clean_participant
from extrapolate import extrapolate_participant
from sample import sample_participant


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Data pipeline")
    parser.add_argument("filenames", type=str, nargs="+", help="filenames of original data")
    parser.add_argument("--clean", action="store_true", help="mark points with pressure = 0 and pen_down = False as cleaned")
    parser.add_argument("--extrapolate", action="store_true", help="extrapolate missing pressure values")
    parser.add_argument("--sample", action="store_true", help="sample missing pressure values")
    parser.add_argument("--pickle", action="store_true", help="write out pickle file")
    parser.add_argument("--concat", action="store_true", help="concatenate to single pickle file")
    args = parser.parse_args()

    dataset = []
    for filename in args.filenames:
        participant = read_original_data(filename)
        if args.clean:
            participant = clean_participant(participant)
        if args.extrapolate:
            participant = extrapolate_participant(participant)
        if args.sample:
            participant = sample_participant(participant)
        if args.pickle:
            write_data(participant, filename + "_preprocessed.pickle")
        else:
            write_original_data(participant, filename + "_preprocessed")
        dataset.append(participant)
    if args.concat:
        write_data(dataset, "data/data.pickle")
