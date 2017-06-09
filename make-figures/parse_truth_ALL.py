"""
Get the groundtruth pathset for an entire file.
Output a dictionary where 'filename instance num' is the key
and the val is set(paths)
"""

import sys
sys.path.insert(0, '../')

from toboggan.parser import enumerate_decompositions
import os
import argparse
from collections import defaultdict


def main(truth_dir, silent=True):
    res = defaultdict(set)

    files = os.listdir(truth_dir)
    for truth_file in files:
        name, extension = truth_file.strip().split('.')
        # we only care about the graph files with name fname
        if extension != 'truth':
            continue
        # open the graph file
        for g_name, g_idx, decomp in enumerate_decompositions(truth_dir + truth_file):
            if not silent:
                print("Processing {}, {}...".format(name, g_idx))
            res[name + " " + str(g_idx)] = set([tuple(x) for x in list(zip(*decomp))[1]])
    
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("truth_dir", type=str,
                        help="directory containing truth files")
    args = parser.parse_args()

    main(args.truth_dir)
