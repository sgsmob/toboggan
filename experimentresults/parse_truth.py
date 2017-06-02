"""Find if a particular set of ground truth solutions have repeated paths."""
from toboggan.parser import enumerate_decompositions
import os
import argparse
from collections import defaultdict


def parse_truth_name(line):
    first_half = line.strip().split(":")[0]
    return list(map(int, first_half.strip().split(" -- ")))


def main(selected_solutions, truth_dir):
    locations = defaultdict(list)
    for line in selected_solutions:
        filenum, name, idx = parse_truth_name(line)
        truth_file = os.path.join(truth_dir, filenum+".truth")
        locations.append((name, idx))

    for truth_file, info in locations.items():
        name, idx = info
        for g_name, g_idx, decomp in enumerate_decompositions(truth_file):
            if g_name == name and g_idx == idx:
                print("Processing {} {} in {}... ".format(idx, name, filenum),
                      end="")
                size = len(decomp)
                unique = set(list(zip(*decomp))[1])
                print("{} repetitions found".format(size-unique))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("selected_solutions", type=argparse.FileType())
    parser.add_argument("truth_dir", type=str)
    args = parser.parse_args()

    main(args.selected_solutions, args.truth_dir)
