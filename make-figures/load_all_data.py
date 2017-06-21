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
    
    all_toboggan_data = {} # collections.defaultdict(dict)
    all_path_info = {'groundtruth':{}, 'toboggan':{}, 'catfish':{}} # collections.defaultdict(dict)

    for which_dataset in data_indices:
        datadirending = datadirs[which_dataset]
        datadir = '/home/kyle/data/rnaseq/' + datadirending + '/'
        froot = froots[which_dataset]
        print("Loading {}".format(froot))

        # [0] Get the rest of toboggan analysis
        inputfile = "all-" + froot + ".txt"
        datadict, datamatrix = table_generator.make_tables(inputfile)
        temp_ddict = {} # collections.defaultdict(list)
        for key, val in datadict.items():
            row = datamatrix[val]
            temp_parts = key.strip().split()
            temp_key = temp_parts[0] + ' ' + temp_parts[1]
            
            if row != list():
                temp_ddict[temp_key] = row
            """
            datamatrix[j] = [ n, m, n_red, m_red,
                           k_groundtruth, cutset_bound, improved_bound, k_opt,
                           t_w, t_path, timeout_flag, timeout_limit, graphname]
            """
        if temp_ddict != list():
            all_toboggan_data[froot] = temp_ddict
            
        # [1] Get ground truth path info
        gt_pathset_dict = parse_truth_all(datadir)
        all_path_info['groundtruth'][froot] = gt_pathset_dict

        # [2] Get catfish path info
        catfish_results_file = '../catfish-comparison/catfish-log-' + froot + '.txt'
        all_catfish_paths = catfish_output_parser(catfish_results_file, gt_pathset_dict, verbose=False)
        all_path_info['catfish'][froot] = all_catfish_paths

        # [3] Get toboggan path info
        toboggan_results_file = './data/master-clean-' + froot + '.txt'
        all_toboggan_paths = toboggan_clean_output_parser(toboggan_results_file, verbose=False)
        # Prune toboggan misfits
        # (these instances somehow ran until successfully terminating, after the timeout limit had elapsed
        temp_list_of_misfits = []
        for key, path in all_toboggan_paths.items():
            if all_toboggan_data[froot][key][10] == '1':
                temp_list_of_misfits.append(key)
        for key in temp_list_of_misfits:
            all_toboggan_paths.pop(key, None)
        all_path_info['toboggan'][froot] = all_toboggan_paths



    print("Done with get_all_data\n")
    return all_path_info, all_toboggan_data