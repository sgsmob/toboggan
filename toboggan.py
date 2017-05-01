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
                print("\tCalling guess_weight with k = {}".format(instance.k))
                solutions = solve(instance, silent=True)
                if bool(solutions):
                    break
                instance.try_larger_k()
            elapsed = time.time() - start
            print("Computation took {:.2f} seconds".format(elapsed))
            print("Solutions:", solutions)
            return solutions
    except TimeoutError:
        print("Timed out after {} seconds".format(maxtime))
        return set()


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
    parser.add_argument('--disprove', help='Run instance with parameter k-1 '
                        'instead of k (needs a .truth file)',
                        action='store_true')
    parser.add_argument("--no_recovery", help="Only print the number of paths"
                        " and their weights in an optimal decomposition, but"
                        " do not recover the paths", action='store_true')

    args = parser.parse_args()

    graph_file = args.file
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

    if not args.skip_truth:
        if path.isfile(truth_file):
            print("# Using ground-truth from file {}".format(truth_file))
    else:
        print("# Not using ground-truth. Guessing parameter.".format(
                truth_file))
        truth_file = None

    # Iterate over every graph-instance inside the input file
    for graphdata, k, index in read_instances(graph_file, truth_file):
        graph, graphname, graphnumber = graphdata

        if instances and index not in instances:
            continue
        start = time.time()
        reduced, mapping = graph.contracted()
        # reduced is the graph after contractions;
        # mapping enables mapping paths on reduced back to paths in graph

        n = len(reduced)
        m = len(list(reduced.edges()))

        if len(reduced) <= 1:
            print("Graph instance named {}:{} is"
                  " trivial.\n".format(graphname, graphnumber))
            continue

        if args.disprove and k:
            k = k - 1
            print("# Using parameter k-1")
        print("Graph instance named {}:{} with n = {}, m = {}, and truth = {}:"
              "".format(graphname, graphnumber,
                        n, m, k if k else "?"), flush=True)

        # create an instance of the graph
        instance = Instance(reduced, k)

        # find the optimal solution size
        solution = find_opt_size(instance, maxtime)
        # recover the paths in an optimal solution
        if bool(solution) and recover:
            weights = solution.pop().path_weights
            start_path_time = time.time()
            print("\tNow recovering the {} paths in the solution {}".format(
                                                                instance.k,
                                                                weights))
            paths = recover_paths(instance, weights)
            elapsed_path_time = time.time() - start_path_time
            print("Path computation took{: .2f} "
                  "seconds".format(elapsed_path_time))
            # Check solution:
            test_flow_cover(reduced, paths)
            print("Paths, weights pass test: flow decomposition confirmed.")

            # Print solutions
            print("Solutions:")
            for path_deq, weight in paths:
                real_path = []
                for arc in path_deq:
                    real_path.extend(mapping[arc])
                node_seq = [graph.source()]
                for arc in real_path:
                    node_seq.append(graph.arc_info[arc]['destin'])
                print("\tPath with weight = {}".format(weight))
                print("\t{}".format(node_seq))
                if args.print_arcs:
                    print("\tarc-labels: {}".format(real_path))

        print()
