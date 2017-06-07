import subprocess as sp
import os
import shutil
import argparse
import time


def iterate_over_instances(graph_file,
                           graph_file_name,
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
                # run catfish on that instance
                starttime = time.time()
                num_paths = run_single_instance(tmp_file_name, path_to_catfish)
                cattime = time.time() - starttime
                os.remove(tmp_file_name)
                # run catfish on that instance
                starttime = time.time()
                ofile = run_single_instance(tmp_file_name, path_to_catfish)
                cattime = time.time() - starttime
                log_line = "# {}\t{}\t{}\t{}\t{}\n".format(graph_file_name,
                                                           instance_name,
                                                           num_paths,
                                                           cattime)
                recording_log_file.write(log_line)
                for ofile_line in ofile:
                    recording_log_file.write(ofile_line)
                ofile.close()
            # figure out the name of the instance
            instance_name = line.strip().split()[-1]
            # make a temporary file for the input
            tmp_string = 'temp_working_file'
            tmp_file_name = tmp_string+".sgr"
            tmp_file = open(tmp_file_name, 'w')

        else:
            tmp_file.write(line)

    tmp_file.close()

    # run catfish on that instance
    starttime = time.time()
    ofile = run_single_instance(tmp_file_name, path_to_catfish)
    cattime = time.time() - starttime
    log_line = "# {}\t{}\t{}\t{}\t{}\n".format(graph_file_name,
                                               instance_name,
                                               num_paths,
                                               cattime)
    recording_log_file.write(log_line)
    for ofile_line in ofile:
        recording_log_file.write(ofile_line)
    ofile.close()

    os.remove(tmp_file_name)


def run_single_instance(input_file_name, path_to_catfish):
    """
    Run Catfish on a single instance and record the paths used
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
    # ofile.close()

    return ofile


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
        # loop through all the instances in the opened files
        truth_file = open(os.path.join(input_dir, name+".truth"), 'r')
        iterate_over_instances(graph_file, name, results_file, graphs_dir,
                               path_to_catfish, recording_log_file)
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
    parser.add_argument("results_file_name", help="name of file to output "
                        " of algorithm", type=str)
    parser.add_argument("-d", "--graphs_dir", nargs="?", default=".",
                        help="place to store output", type=str)
    parser.add_argument("-c", "--catfish", nargs="?", default="/home/kakloste/"
                        "toboggan/catfish/src/src/catfish", help="full path to"
                        " catfish executable", type=str)
    parser.add_argument("recording_log_name", help="name of file to store"
                        " notes of timing etc", type=str)
    args = parser.parse_args()
    main(args)
