"""
Get the groundtruth pathset for a list of instances,
check if the pathsets contain repetitions.
"""

import sys
sys.path.insert(0, '../')

from toboggan.parser import enumerate_decompositions
import os
import argparse
from collections import defaultdict


def main(selected_solutions, truth_dir):
    locations = defaultdict(list)
    res = defaultdict(dict)
    for line in selected_solutions:
        filenum = line.strip().split(".")[0]
        instancenum = line.strip().split()[1]
        truth_file = os.path.join(truth_dir, filenum+".truth")
        locations[truth_file].append(instancenum)

    print("====================================================")
    print("Input list parsed, beginning processing of data")
    print("====================================================")
    for truth_file, instance_list in locations.items():
        filename = truth_file.strip().split('/')[-1].split('.')[0] + ".graph"
        for g_name, g_idx, decomp in enumerate_decompositions(truth_file):
            for idx in instance_list:
                if g_idx == idx:
                    print("Processing {}, {}...".format(idx, filenum), end="")
                    size = len(decomp)
                    unique = set([tuple(x) for x in list(zip(*decomp))[1]])
                    print("{} repetitions found".format(size-len(unique)))
                    
                    decomp_path_weight_dict = defaultdict(list)
                    for j in range(len(decomp)):
                        this_path = tuple(decomp[j][1])
                        this_weight = decomp[j][0]
                        decomp_path_weight_dict[this_path].append(this_weight)
            
                    res[filename + " " + str(idx)] = {
                                                        'num_repetitions':size-len(unique),
                                                        'unique_paths_set':unique,
                                                        'decomp':decomp_path_weight_dict
                                                     }
    
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("selected_solutions", type=argparse.FileType(),
                        help="list of files to iterate over")
    parser.add_argument("truth_dir", type=str,
                        help="directory containing truth files")
    args = parser.parse_args()

    main(args.selected_solutions, args.truth_dir)
