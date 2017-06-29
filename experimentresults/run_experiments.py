import os
import argparse
import subprocess


def iterate_over_directory(input_dir,
                           results_file,
                           timeout):
    # get all the files in the directory
    files = os.listdir(input_dir)
    for f in files:
        name, extension = os.path.splitext(f)
        # we only care about the graph files
        if extension != ".graph":
            continue
        # open the graph file
        print("Processing file {}".format(f))
        subprocess.call("python3 ../toboggan.py {}/{} --skip_truth "
                        "--experiment_info --timeout {} >>"
                        " {}".format(input_dir, f, timeout, results_file),
                        shell=True)


def main(args):
    timeout = -1
    if args.timeout:
        timeout = args.timeout
    iterate_over_directory(args.input_dir, args.results_file_name, timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="directory containing .graph and"
                        ".truth files", type=str)
    parser.add_argument("results_file_name", help="name of file to store notes"
                        "of instances that are nonoptimal", type=str)
    parser.add_argument("timeout", help="time to run before skipping instance",
                        type=int)

    args = parser.parse_args()
    main(args)
