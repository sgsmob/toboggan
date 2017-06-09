from collections import defaultdict
import numpy

def process_algorithm_output(gt_pathset_dict, all_algorithm_paths):
    def compute_stats(set_1, set_2):
        n = len(set_1.intersection(set_2))
        jaccard = n / float(len(set_1) + len(set_2) - n)
        precision = 0
        if len(set_1) > 0:
            precision = n/len(set_1)
        return jaccard, precision

    all_stats = []
    gt_different_from_algorithm = []
    gt_identical = []
    indices_timeout = []

    num_identical = 0
    num_trivial = 0
    num_timeouts = 0
    for key, gt_pathset in gt_pathset_dict.items():
        if len(gt_pathset) == 1:
            num_trivial += 1
            continue
        if key not in all_algorithm_paths:
            if len(gt_pathset) > 1:
                num_timeouts += 1
                indices_timeout.append(len(all_stats) - 1)
        algorithm_pathset = all_algorithm_paths[key]
        gap = len(gt_pathset) - len(algorithm_pathset)
        this_jaccard_value, this_intersect = compute_stats(algorithm_pathset, gt_pathset)
        all_stats.append( [len(gt_pathset), len(algorithm_pathset), this_jaccard_value, this_intersect, gap] )

        if this_jaccard_value == 1 and gap == 0:
            gt_identical.append(len(all_stats) - 1)
            num_identical += 1
        else:
            gt_different_from_algorithm.append( len(all_stats) - 1 )

    print("Num timeouts = {}".format(num_timeouts))
    return numpy.matrix(all_stats), gt_identical, gt_different_from_algorithm, indices_timeout


def catfish_output_parser(catfish_results_file, filename_instancenum_dict, verbose=False):
    
    def convert_text_to_path(line):
            text_list = line.strip().split('\t')[1]
            text_list = text_list[1:-1].split(', ')  # omit the brackets and commas
            return list(map(lambda x: int(x), text_list))
        
    # Make dict with key = filename, val = set of instances in that file to check
    key_dict = filename_instancenum_dict.copy()

    solutions = {}
    current_soln_key = None
    paths_found = False
    instance_found = False
    catfish_path_dict = defaultdict(list)
    catfish_weight = []
    key_pattern = None

    # iterate over lines to the end of file,
    # checking for instances in instance_set
    with open(catfish_results_file, 'r') as reader:
        for line in reader:
            # get filename+instancenum info
            current_line = line.strip().split()
            if '#' in line:
                current_key_pattern = current_line[1] + ' ' + current_line[2]  # key = 'filenum instancenum'
                catfish_path_dict = defaultdict(list)
                
            elif "path" in line:
                path_weight = current_line[4]
                path = list(map(lambda x: int(x), current_line[7::]))
                catfish_path_dict[ tuple(path) ].append(path_weight)
                solutions[current_key_pattern] = set( catfish_path_dict.keys() )

    return solutions



def toboggan_output_parser(toboggan_results_file, filename_instancenum_dict, verbose=False):
    
    def convert_text_to_path(line):
            text_list = line.strip().split('\t')[1]
            text_list = text_list[1:-1].split(', ')  # omit the brackets and commas
            return list(map(lambda x: int(x), text_list))
        
    # Make dict with key = filename, val = set of instances in that file to check
    key_dict = filename_instancenum_dict.copy()

    solutions = defaultdict(set)
    current_soln_key = None
    paths_found = False
    instance_found = False
    toboggan_path_dict = defaultdict(list)
    toboggan_weight = []
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
                if line.startswith("#") and "Solutions:" in line and paths_found is False:
                    if verbose: print("Found solution line")
                    paths_found = True
                if paths_found is True:
                    if 'weight = ' in line:
                        toboggan_weight = int(line.strip().split()[-1])
                    elif'[' in line:
                        path = convert_text_to_path(line)
                        toboggan_path_dict[ tuple(path) ].append(toboggan_weight)
            if 'Finished instance.' in line and (instance_found and paths_found):
                solutions[current_soln_key] = set( toboggan_path_dict.keys() )
                current_soln_key = None
                key_pattern = None
                instance_found = False
                paths_found = False
                toboggan_path_dict = defaultdict(list)
                toboggan_weight = []
    return solutions