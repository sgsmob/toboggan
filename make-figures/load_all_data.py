# [1] COLLECT ALL GROUND TRUTH DATA
from parse_truth_ALL import main as parse_truth_all
import collections
from plot_overlap_ratios import plot_ratio_success
from algorithm_output_parser import process_algorithm_output
from algorithm_output_parser import toboggan_output_parser
from algorithm_output_parser import catfish_output_parser

import table_generator


def get_all_data(data_indices=[0,1,2]):
    froots = ['human', 'mouse', 'zebra']
    datadirs = ['human', 'mouse', 'zebrafish']
    
    all_toboggan_data = collections.defaultdict(dict)
    all_path_info = collections.defaultdict(dict)

    for which_dataset in data_indices:
        datadirending = datadirs[which_dataset]
        datadir = '/home/kyle/data/rnaseq/' + datadirending + '/'
        froot = froots[which_dataset]
        print("Loading {}".format(froot))

        # [1] Get ground truth path info
        gt_pathset_dict = parse_truth_all(datadir)

        # [2] Get catfish path info
        catfish_results_file = '../catfish-comparison/catfish-log-' + froot + '.txt'
        all_catfish_paths = catfish_output_parser(catfish_results_file, gt_pathset_dict, verbose=False)

        # [3] Get toboggan path info
        toboggan_results_file = './data/' + froot + '-master-file.txt'
        all_toboggan_paths = toboggan_output_parser(toboggan_results_file, gt_pathset_dict, verbose=False)

        # [4] Get the rest of toboggan analysis
        inputfile = "all-" + froot + ".txt"
        datadict, datamatrix, _, _ = table_generator.make_tables(inputfile)
        temp_ddict = collections.defaultdict(list)
        for key, val in datadict.items():
            row = datamatrix[val]
            temp_parts = key.strip().split()
            temp_key = temp_parts[0].split('.')[0] + ' ' + temp_parts[1]
            temp_ddict[temp_key] = row
            """
            datamatrix[j] = [ n, m, n_red, m_red,
                              k_groundtruth, k_opt, cutset_bound, improved_bound,
                              t_w, t_path]
            """
        all_toboggan_data[froot] = temp_ddict
        all_path_info['groundtruth'][froot] = gt_pathset_dict
        all_path_info['toboggan'][froot] = all_toboggan_paths
        all_path_info['catfish'][froot] = all_catfish_paths

    print("Done with get_all_data\n")
    return all_path_info, all_toboggan_data