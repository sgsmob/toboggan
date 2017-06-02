"""
Given an output file from run_experiment.py, this script aggregates the relevant info
and compiles into a table, one row at a time, and prints to screen

Use:
    python3 file_scrape.py [experimentresults file]

    - the experimentresults file should contain output from run_experiments.py
"""

from collections import Counter
import argparse


def main(args):
    datafile = args.datafile
    # print("# fname\t inst\t n\t m\t n_red\t m_red\t k_GT\t k_opt\t cut_bd\t impr_bd\t time_w\t time_p\t time_out")

    files = []
    instance_num = []
    name = []

    timed_out_graphs = []

    n_in = []
    m_in = []
    n_red = []
    m_red = []
    k_gtrue = []
    k_cutset = []
    k_improved = []
    k_opt = []
    time_w = []
    time_p = []
    time_out = []
    weights = []

    time_flag = 0
    timeout_limits = []

    with open(datafile, 'r') as reader:
        content = reader.readlines()
        for line in content:
            if line[0] == "#":  # skip that line
                continue
            else:
                parts = line.strip().split()
                if(len(parts) > 0):
                    if(parts[0] == 'File'):
                        tmp_file = parts[1]
                        tmp_inst = parts[3]
                        tmp_name = parts[5]
                        files.append(tmp_file)
                        instance_num.append(tmp_inst)
                        name.append(tmp_name)
                        timeout_limits.append(-1)
                        tmp_timeout_limit = -1
                    if(parts[0] == 'Searching'):
                        tmp_timeout_limit = parts[-1]
                    if(parts[0] == 'Timed'):
                        timed_out_graphs.append((tmp_file, tmp_inst, tmp_name))
                        time_flag = 1
                    if(parts[0] == 'All_info'):
                        n_in.append(parts[1])
                        m_in.append(parts[2])
                        n_red.append(parts[3])
                        m_red.append(parts[4])
                        k_gtrue.append(parts[5])
                        k_cutset.append(parts[6])
                        k_improved.append(parts[7])
                        k_opt.append(parts[8])
                        time_w.append(parts[9])
                        time_p.append(parts[10])
                    if(parts[0] == 'Finished'):
                        time_out.append(time_flag)
                        time_flag = 0
                        timeout_limits[-1] = tmp_timeout_limit
    for j in range(len(n_in)):
        # print("# fname\t inst\t n\t m\t n_red\t m_red\t k_GT\t k_opt\t cut_bd\t impr_bd\t time_w\t time_p\t time_out")
        tmp_file = files[j]
        tmp_inst = instance_num[j]
        namej = name[j]
        n_inj = n_in[j]
        m_inj = m_in[j]
        n_redj = n_red[j]
        m_redj = m_red[j]
        k_gtruej = k_gtrue[j]
        k_optj = k_opt[j]
        k_cutsetj = k_cutset[j]
        k_improvedj = k_improved[j]
        time_wj = float(time_w[j])
        time_pj = float(time_p[j])
        time_outj = time_out[j]
        timeout_val = timeout_limits[j]

        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10f}\t{:.10f}\t{}\t{}\t{}".format(
               tmp_file, tmp_inst, n_inj, m_inj, n_redj, m_redj, k_gtruej, k_optj,
                k_cutsetj, k_improvedj, time_wj, time_pj, time_outj, timeout_val, namej))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="file containing output"
                        "from toboggan experiment", type=str)
    args = parser.parse_args()
    main(args)