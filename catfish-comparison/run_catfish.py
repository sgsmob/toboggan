import subprocess as sp
import os
import shutil
import argparse
import time


def iterate_over_instances(graph_file,
                           truth_file,
                           graph_file_name,
                           truth_lookup,
                           results_file,
                           graphs_dir,
                           path_to_catfish,
                           recording_log_file):
    """
    Iterate over the graph file and run on each instance
    """
    tmp_file_name = None
    tmp_file = None
    # the truth file starts with a '#' that we want to discount
    # _ = truth_file.readline()

    for line in graph_file:
        if line[0] == "#":  # we've reached a new instance
            # clean up when not first iteration
            if tmp_file is not None:
                tmp_file.close()
                tmp_truth_file.close()
                # run catfish on that instance
                starttime = time.time()
                num_paths = run_single_instance(tmp_file_name, path_to_catfish)
                cattime = time.time() - starttime
                # check if we got the optimal number of paths
                check_if_optimal(num_paths,
                                 true_num_paths,
                                 results_file,
                                 graph_file_name,
                                 instance_name,
                                 tmp_file_name,
                                 tmp_truth_file_name,
                                 graphs_dir)
                os.remove(tmp_file_name)
                os.remove(tmp_truth_file_name)
                log_line = "{}\t{}\t{}\t{}\t{}\n".format(graph_file_name,
                                                       instance_name,
                                                       true_num_paths,
                                                       num_paths,
                                                       cattime)
                recording_log_file.write(log_line)
            # figure out the name of the instance
            instance_name = line.strip().split()[-1]
            # count the number of paths in the corresponding segment of the
            # truth file
            true_num_paths = truth_lookup[instance_name]
            # make a temporary file for the input
            tmp_string = 'temp_working_file'
            tmp_file_name = tmp_string+".sgr"
            tmp_file = open(tmp_file_name, 'w')

            tmp_truth_file_name = tmp_string+".sgrtruth"
            tmp_truth_file = open(tmp_truth_file_name, 'w')
            truth_write_flag = False
            truth_file.seek(0)
            # grabbing the .truth info for the graph instance
            for truth_line in truth_file:
                if truth_line[0] == '#':
                    if truth_write_flag is True:
                        break
                    truth_instance = truth_line.strip().split()[-1]
                    # print(instance_name)
                    if truth_instance == instance_name:
                        truth_write_flag = True
                        # print(graph_file_name + ': flag is true\n')
                    else:
                        continue
                else:
                    if truth_write_flag is False:
                        continue
                    else:
                        tmp_truth_file.write(truth_line)
                        # print(truth_line)

        else:
            tmp_file.write(line)

    tmp_file.close()
    tmp_truth_file.close()

    # run catfish on that instance
    starttime = time.time()
    num_paths = run_single_instance(tmp_file_name, path_to_catfish)
    cattime = time.time() - starttime
    log_line = "{}\t{}\t{}\t{}\t{}\n".format(graph_file_name,
                                           instance_name,
                                           true_num_paths,
                                           num_paths,
                                           cattime)
    recording_log_file.write(log_line)
    # check if we got the optimal number of paths
    check_if_optimal(num_paths,
                     true_num_paths,
                     results_file,
                     graph_file_name,
                     instance_name,
                     tmp_file_name,
                     tmp_truth_file_name,
                     graphs_dir)
    os.remove(tmp_file_name)
    os.remove(tmp_truth_file_name)


def run_single_instance(input_file_name, path_to_catfish):
    """
    Run Catfish on a single instance and return the number of paths used
    """

    output_file_name = os.path.join(os.getcwd(), "output.out")
    """
    args = [path_to_catfish, "-i", input_file_name, "-o", output_file_name]
    proc = sp.Popen(args, stdout=sp.PIPE)
    stdout, stderr = proc.communicate()
    """
    sp.call("{} -i {} -o {}".format(path_to_catfish, input_file_name,
                                 output_file_name), shell=True)
    # print "Output:"
    # print stdout
    # print "Errors:"
    # print stderr
    # find the number of paths used
    ofile = open(output_file_name, 'r')
    num_paths = count_paths(ofile)
    ofile.close()
    return num_paths


def count_paths(paths_file):
    """
    Count the number of paths in the file, stopping at the next '#' if present
    """
    count = 0
    for line in paths_file:
        if line[0] == "#":
            break
        else:
            count += 1
    return count


def create_truth_lookup(truth_file):
    count = 0
    table = {}
    idx = None
    for line in truth_file:
        if line[0] == "#":
            if idx is not None:
                table[idx] = count
                count = 0
            idx = line.strip().split()[-1]
        else:
            count += 1

    table[idx] = count
    return table


def check_if_optimal(num_paths,
                     true_num_paths,
                     results_file,
                     graph_file_name,
                     instance_name,
                     tmp_file_name,
                     tmp_truth_name,
                     graphs_dir):
    """
    Determine whether Catfish returned an optimal solution.  If not, note the
    name and result of that output and keep the file name.
    """
    if num_paths > true_num_paths:
        # make the line of data
        line = "{},{},{}\n".format(graph_file_name+"."+instance_name,
                                   num_paths, true_num_paths)
        results_file.write(line)
        # copy the temporary graph
        perm_file_name = os.path.join(graphs_dir,
                                      "{}.{}.sgr".format(graph_file_name,
                                                         instance_name))
        shutil.copy(tmp_file_name, perm_file_name)
        # copy the temporary truth file
        perm_file_name = os.path.join(graphs_dir,
                                      "{}.{}.sgrtruth".format(graph_file_name,
                                                              instance_name))
        shutil.copy(tmp_truth_name, perm_file_name)
        print(perm_file_name + ' ' + tmp_file_name + ' ' + tmp_truth_name)


def iterate_over_directory(input_dir,
                           results_file,
                           graphs_dir,
                           path_to_catfish,
                           recording_log_file):
    # get all the files in the directory
    files = os.listdir(input_dir)
    for f in files:
        name, extension = os.path.splitext(f)
        # we only care about the graph files
        if extension != ".graph":
            continue
        # open the graph file
        print("Processing file {}".format(f))
        graph_file = open(os.path.join(input_dir, f), 'r')
        # the truth file has the same name but different extension
        truth_file = open(os.path.join(input_dir, name+".truth"), 'r')
        truth_lookup = create_truth_lookup(truth_file)
        truth_file.close()
        # loop through all the instances in the opened files
        truth_file = open(os.path.join(input_dir, name+".truth"), 'r')
        iterate_over_instances(graph_file, truth_file, name, truth_lookup,
                               results_file, graphs_dir, path_to_catfish,
                               recording_log_file)
        graph_file.close()
        truth_file.close()


def main(args):
    input_dir = args.input_dir
    results_file = open(args.results_file_name, 'w')
    graphs_dir = args.graphs_dir
    path_to_catfish = args.catfish
    recording_log_file = open(args.recording_log_name, 'w')
    iterate_over_directory(input_dir, results_file, graphs_dir,
                           path_to_catfish, recording_log_file)
    results_file.close()
    recording_log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="directory containing .graph and "
                        ".truth files", type=str)
    parser.add_argument("results_file_name", help="name of file to store notes"
                        " of instances that are nonoptimal", type=str)
    parser.add_argument("-d", "--graphs_dir", nargs="?", default=".",
                        help="place to store graphs where catfish is"
                        " nonoptimal", type=str)
    parser.add_argument("-c", "--catfish", nargs="?", default="/home/kakloste/"
                        "toboggan/catfish/src/src/catfish", help="full path to"
                        " catfish executable", type=str)
    parser.add_argument("recording_log_name", help="name of file to store notes"
                        " of timing etc", type=str)
    args = parser.parse_args()
    main(args)
