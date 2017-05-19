import os
import shutil
import argparse
import subprocess


def iterate_over_input_list(input_list,
                            input_dir,
                            results_file,
                            timeout):
    with open(input_list, 'r') as instancelist:
        for line in instancelist:
            print(line.strip())
            if line[0] == "#":  # skip that line
                continue
            parts = line.strip().split()
            if(len(parts) > 0):
                filename = parts[0]
                instancenum = parts[1]
                instancenum = int(instancenum) + 1

                subprocess.call("python3 ../toboggan.py {}/{} --indices {} "
                    "--skip_truth --experiment_info --timeout {} >> {}"
                    "".format(input_dir, filename, instancenum, timeout, results_file), shell=True)


def main(args):
    timeout = -1
    if args.timeout:
        timeout = args.timeout
    iterate_over_input_list(args.input_list, args.input_dir,
                            args.results_file_name, timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", help="directory containing .graph and"
                        ".truth files", type=str)
    parser.add_argument("input_list", help="file containing list of names "
                        "of files and instances to run toboggan on.", type=str)
    parser.add_argument("results_file_name", help="name of file to store notes"
                        "of instances that are nonoptimal", type=str)
    parser.add_argument("timeout", help="time to run before skipping instance",
                        type=int)

    args = parser.parse_args()
    main(args)
