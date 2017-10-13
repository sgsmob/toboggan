"""
    This script scans the output.txt file output from toboggan,
    and creates a dictionary with a key for each graph instance,
    with value = the set of path-weights in the toboggan solution
    for that graph instance.

    Note: this skips trivial instances and timeouts.
"""

import argparse
from collections import defaultdict


def toboggan_output_weights_parser(toboggan_results_file):

    def convert_text_to_path(line):
            text_list = line.strip().split('\t')[1]
            text_list = text_list[1:-1].split(', ')
            return list(map(lambda x: int(x), text_list))

    solutions = defaultdict(set)
    current_soln_key = None
    paths_found = False
    instance_found = False
    toboggan_weights = []
    key_pattern = None

    # iterate over lines to the end of file,
    # checking for instances in instance_set
    with open(toboggan_results_file, 'r') as reader:
        for line in reader:
            # get filename+instancenum info
            if 'File' in line and 'instance' in line and 'name' in line:
                temp_line = line.strip().split()
                temp_fname = temp_line[1].split('.')[0]
                key_pattern = temp_fname + ' ' + temp_line[3]
                instance_found = True
                current_soln_key = key_pattern

            elif instance_found is True:
                if line.startswith("#") and "Solutions:" in line \
                   and paths_found is False:
                    paths_found = True
                if paths_found is True:
                    if 'weight = ' in line:
                        toboggan_weights.append(int(line.strip().split()[-1]))

            if 'Finished instance.' in line \
               and (instance_found and paths_found):
                solutions[current_soln_key] = set(toboggan_weights)
                current_soln_key = None
                key_pattern = None
                instance_found = False
                paths_found = False
                toboggan_weights = []
    return solutions


def main(args):

    toboggan_results_file = args.datafile
    processed_output_file = args.output
    toboggan_paths = toboggan_output_weights_parser(toboggan_results_file)
    with open(processed_output_file, 'w') as outputfile:
        for key, weightset in toboggan_paths.items():
            line = key
            for weight in weightset:
                line = line + ' ' + str(weight)
            outputfile.write(line + ' \n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="file containing output"
                        "from toboggan experiment", type=str)
    parser.add_argument("output", help="file to write to", type=str)
    args = parser.parse_args()
    main(args)
