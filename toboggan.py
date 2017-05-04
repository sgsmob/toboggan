#! /usr/bin/env python3
#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# -*- coding: utf-8 -*-
# python libs
import time
import argparse
import signal

# local imports
from os import path
from toboggan.guess_weight import solve
from toboggan.parser import read_instances
from toboggan.flow import Instance
from toboggan.dp import recover_paths
from toboggan.graphs import test_flow_cover


# Timeout context, see
#   http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
class timeout:
    """
    Enable a timeout for a function call.

    Used to skip an input graph-instance once our algorithm has run for a
    specified amount of time without terminating.
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if self.seconds > 0:
            signal.alarm(0)


def index_range(raw):
    """Parse index ranges separated by commas e.g. '1,2-5,7-11'."""
    if not raw:
        return None

    indices = set()
    try:
        with open(raw, 'r') as f:
            for l in f:
                l = l.strip()
                if len(l) == 0 or l[0] == '#':
                    continue
                if " " in l:
                    l = l.split()[0]
                indices.add(int(l))
        return indices
    except FileNotFoundError:
        pass

    for s in raw.split(","):
        if "-" in s:
            start, end = s.split("-")
            indices.update(range(int(start), int(end)+1))
        else:
            indices.add(int(s))
    return indices


def find_opt_size(instance, maxtime):
    """Find the optimum size of a flow decomposition."""
    if maxtime is None:
        maxtime = -1
    try:
        with timeout(seconds=maxtime):
            while True:
                print("# \tCall guess_weight with k = {}".format(instance.k))
                solutions = solve(instance, silent=True)
                if bool(solutions):
                    break
                instance.try_larger_k()
            elapsed = time.time() - start
            print("# Weights computation took {:.2f} seconds".format(elapsed))
            print("# Solution:", solutions)
            return solutions, elapsed
    except TimeoutError:
        print("Timed out after {} seconds".format(maxtime))
        return set(), maxtime


if __name__ == "__main__":
    """
        Main script
    """
    parser = argparse.ArgumentParser(description='Splice flows into paths')
    parser.add_argument('file', help='a .graph file.'
                        ' Needs a .truth file in the same folder.')
    parser.add_argument('--indices',
                        help='Either a file containing indices '
                        '(position in .graph file) on which to run, '
                        'or a list of indices separated by commas. '
                        'Ranges are accepted as well, e.g. "1,2-5,6"',
                        type=index_range)
    parser.add_argument('--timeout', help='Timeout in seconds', type=int)
    parser.add_argument('--skip_truth', help="Do not check for *.truth."
                        " Instead, start from our computed lower-bound on k.",
                        action='store_true')
    parser.add_argument('--print_arcs', help="Make output include arc labels.",
                        action='store_true')
    parser.add_argument('--print_contracted', help="Print contracted graph.",
                        action='store_true')
    parser.add_argument('--disprove', help='Run instance with parameter k-1 '
                        'instead of k (needs a .truth file)',
                        action='store_true')
    parser.add_argument('--experiment_info', help='Print out experiment-'
                        'relevant info in format convenient for processing.',
                        action='store_true')
    parser.add_argument("--no_recovery", help="Only print the number of paths"
                        " and their weights in an optimal decomposition, but"
                        " do not recover the paths", action='store_true')

    args = parser.parse_args()

    graph_file = args.file
    filename = graph_file.split("/")[-1]
    tokens = path.basename(graph_file).split(".")
    tokens[-1] = ".truth"
    truth_file = path.join(path.dirname(graph_file), "".join(tokens))

    maxtime = args.timeout
    if maxtime:
        print("# Timeout is set to", maxtime)
    else:
        print("# No timeout set")

    recover = not args.no_recovery
    if recover:
        print("# Recovering paths")
    else:
        print("# Only printing path weights")

    instances = args.indices
    if instances:
        a = sorted(list(instances))
        res = str(a[0])
        lastprinted = a[0]
        for current, last in zip(a[1:], a[:-1]):
            if current == last+1:
                continue
            if last != lastprinted:
                res += "-"+str(last)
            res += ","+str(current)
            lastprinted = current

        if lastprinted != a[-1]:
            res += "-" + str(a[-1])

        print("# Running on instances", res)
    else:
        print("# Running on all instances")

    if path.isfile(truth_file):
        print("# Ground-truth available in file {}".format(truth_file))
    else:
        print("# No ground-truth available. Guessing parameter.".format(
                truth_file))
        truth_file = None

    # Iterate over every graph-instance inside the input file
    for graphdata, k, index in read_instances(graph_file, truth_file):
        graph, graphname, graphnumber = graphdata

        if instances and index not in instances:
            continue
        n_input = len(graph)
        m_input = len(list(graph.edges()))
        k_gtrue = k if k else "?"
        k_opt = None
        k_improve = 0
        weights = []
        time_weights = None
        time_path = None
        print("\nFile {} instance {} name {} with n = {}, m = {}, and truth = "
              "{}:".format(filename, graphnumber, graphname, n_input,
                           m_input, k if k else "?"), flush=True)
        start = time.time()
        reduced, mapping = graph.contracted()
        # reduced is the graph after contractions;
        # mapping enables mapping paths on reduced back to paths in graph
        if args.print_contracted:
            reduced.print_out()

        n = len(reduced)
        m = len(list(reduced.edges()))

        if len(reduced) <= 1:
            print("Trivial.")
            # continue
            k_improve = 1
            k_opt = 1
            k_cutset = 1
            time_weights = 0
            time_paths = 0
            if m_input != 0:
                weights = [list(graph.edges())[0][2]]
            else:
                weights = [0]
        else:
            if args.disprove and k:
                k = k - 1
                print("# Using parameter k-1")

            # create an instance of the graph
            if args.skip_truth:
                k = 1
            instance = Instance(reduced, k)
            k_improve = instance.best_cut_lower_bound
            print("# Reduced instance has n = {}, m = {}, and lower_bound "
                  "= {}:".format(n, m, instance.k), flush=True)

            k_cutset = instance.max_edge_cut_size

            # find the optimal solution size
            solution_weights, time_weights = find_opt_size(instance, maxtime)
            # recover the paths in an optimal solution
            if bool(solution_weights) and recover:
                weights = solution_weights.pop().path_weights
                start_path_time = time.time()
                print("#\tNow recovering the {} paths in the solution {}"
                      "".format(instance.k, weights))
                solution_paths = recover_paths(instance, weights)
                time_paths = time.time() - start_path_time
                print("# Recovery took{: .2f} "
                      "seconds".format(time_paths))
                # Check solution:
                test_flow_cover(reduced, solution_paths)
                print("# Paths, weights pass test: flow decomposition"
                      " confirmed.")
                # Print solutions
                print("# Solutions:")
                weight_vec = []
                k_opt = len(weights)
                for path_deq, weight in solution_paths:
                    real_path = []
                    for arc in path_deq:
                        real_path.extend(mapping[arc])
                    node_seq = [graph.source()]
                    for arc in real_path:
                        node_seq.append(graph.arc_info[arc]['destin'])
                    print("# \tPath with weight = {}".format(weight))
                    weight_vec.append(weight)
                    print("# \t{}".format(node_seq))
                    if args.print_arcs:
                        print("\tarc-labels: {}".format(real_path))

        if args.experiment_info:
            print("# All_info\tn_in\tm_in\tn_red\tm_red\tk_gtrue\tk_cut"
                  "\tk_impro\tk_opt\ttime_w\ttime_p")
            print("All_info\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                  "".format(n_input, m_input, n, m, k_gtrue, k_cutset,
                            k_improve, k_opt, time_weights,
                            time_paths))
            print("weights\t", *[w for w in weights])
        print("Finished instance.\n")
