"""
Calls file_scrape on all files of a single type, i.e. human/zebra/mouse
outputs the relevant info into a single txt file
"""

import argparse
import os
import shutil
import subprocess


def iterate_over_directory(input_dir, fname, results_file):
    # get all the files in the directory
    files = os.listdir(input_dir)
    for f in files:
        name, extension = os.path.splitext(f)
        # we only care about the graph files with name fname
        if name.split('-')[0] != 'output':
            continue
        if name.split('-')[1] != fname:
            continue
        # open the graph file
        print("Processing file {}".format(f))
        subprocess.call("python3 file_scrape.py {}/{} >> {}".format(input_dir, f, results_file), shell=True)


def main(args):
    iterate_over_directory(args.input_dir, args.fname, args.results_file_name)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="directory containing the output-[dataname].txt"
                        " files", type=str)
    parser.add_argument("fname", help="name of data type: human, zebra, mouse",
                        type=str)
    parser.add_argument("results_file_name", help="name of file to store notes"
                        "of instances that are nonoptimal", type=str)

    args = parser.parse_args()
    main(args)