"""
    Modified from original toboggan toa ssume only preprocessing routines are running,
    no path decomposition components run.
"""
import os
import argparse
import subprocess


def iterate_over_directory(input_dir,
                           results_file):
    # get all the files in the directory
    files = os.listdir(input_dir)
    for f in files:
        name, extension = os.path.splitext(f)
        # we only care about the graph files
        if extension != ".graph":
            continue
        # open the graph file
        print("Processing file {}".format(f))
        subprocess.call("python3 toboggan.py {}/{} --skip_truth >> "
                        " {}".format(input_dir, f, results_file),
                        shell=True)


def main(args):
    iterate_over_directory(args.input_dir, args.results_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="directory containing .graph and"
                        " .truth files", type=str)
    parser.add_argument("results_file_name", help="name of file to store notes"
                        " of instances on which toboggan times out", type=str)

    args = parser.parse_args()
    main(args)
