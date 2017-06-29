"""
Given an output file from run_experiment.py, this script aggregates the
relevant info and compiles into a table, one row at a time, and prints
to screen.

Usage:
    python3 file_scrape.py [output.txt]

    - the file output.txt should contain output from run_experiments.py
"""

import argparse


def main(args):
    datafile = args.datafile
    timeout_flag = 0
    timeout_limit = -1
    single_instance_info = [None for x in range(15)]

    """
    For each graph instance, read through toboggan's output and construct as
    follows:

    [filename, instance num,
     n_input, m_input, n_reduced, m_reduced,
     k_groundtruth, cutset_bound, improved_bound, k_optimal,
     time_weights, time_path, timeout_flag, timeout_limit,
     graphname]
    """

    with open(datafile, 'r') as reader:
        content = reader.readlines()
        for line in content:
            if line[0] == "#":  # skip that line
                continue
            else:
                parts = line.strip().split()
                if(len(parts) > 0):
                    if(parts[0] == 'File'):
                        single_instance_info = [None for x in range(15)]
                        single_instance_info[0] = parts[1]
                        single_instance_info[1] = parts[3]
                        single_instance_info[14] = parts[5]
                        timeout_limit = -1
                        timeout_flag = 0
                    if(parts[0] == 'Searching'):
                        timeout_limit = parts[-1]
                    if(parts[0] == 'Timed'):
                        timeout_flag = 1
                    if(parts[0] == 'All_info'):
                        single_instance_info[2] = parts[1]
                        single_instance_info[3] = parts[2]
                        single_instance_info[4] = parts[3]
                        single_instance_info[5] = parts[4]
                        single_instance_info[6] = parts[5]
                        single_instance_info[7] = parts[6]
                        single_instance_info[8] = parts[7]
                        single_instance_info[9] = parts[8]
                        single_instance_info[10] = float(parts[9])
                        single_instance_info[11] = float(parts[10])

                    if(parts[0] == 'Finished'):
                        single_instance_info[12] = timeout_flag
                        single_instance_info[13] = timeout_limit
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                              "\t{:.10f}\t{:.10f}\t{}\t{}\t{}"
                              "".format(*single_instance_info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="file containing output"
                        "from toboggan experiment", type=str)
    args = parser.parse_args()
    main(args)
