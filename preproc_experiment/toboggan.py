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
import cProfile
# local imports
from os import path
import sys
from toboggan.parser import read_instances
from toboggan.flow import Instance


# Override error message to show help message instead
class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        sys.exit(2)


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

if __name__ == "__main__":
    """
        Main script
    """
    with open('readme_short.txt', 'r') as desc:
        description = desc.read()
    parser = DefaultHelpParser(description=description)
    parser.add_argument('file', help='A .graph file containing the input graph(s).')
    parser.add_argument('--indices', help='Either a file containing indices '
                        '(position in .graph file) on which to run, '
                        'or a list of indices separated by commas. '
                        'Ranges are accepted as well, e.g. "1,2-5,6".',
                        type=index_range)
    parser.add_argument('--timeout', help='Timeout in seconds, after which execution'
                        ' for a single graph will be stopped.', type=int)
    parser.add_argument('--skip_truth', help="Do not check for *.truth."
                        " Instead, start from our computed lower-bound on k.",
                        action='store_true')

    args = parser.parse_args()

    graph_file = args.file
    filename = path.basename(graph_file)
    truth_file = "{}.truth".format(path.splitext(graph_file)[0])

    maxtime = args.timeout


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


    if not path.isfile(truth_file):
        truth_file = None

    # Iterate over every graph-instance inside the input file
    for graphdata, k, index in read_instances(graph_file, truth_file):
        graph, graphname, graphnumber = graphdata

        if instances and index not in instances:
            continue
        n_input = len(graph)
        m_input = len(list(graph.edges()))
        k_gtrue = k if k else "?"
        k_improve = 0
        k_cutset = 0

        # TIME THE CONTRACTION OPERATION
        start = time.time()
        reduced, mapping = graph.contracted()
        time_contract = time.time() - start

        n = len(reduced)
        m = len(list(reduced.edges()))

        # TIME THE CUTSET ANALYSIS
        start = time.time()
        if len(reduced) <= 1:
            k_improve = 1
            k_cutset = 1
        else:
            # create an instance of the graph
            if args.skip_truth:
                k = 1
            instance = Instance(reduced, k)
            k_improve = instance.best_cut_lower_bound
            k_cutset = instance.max_edge_cut_size

        time_cutset = time.time() - start


        # print experimental statistics
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format( filename, graphnumber,
                             n_input, m_input, n, m,
                             k_gtrue, k_cutset, k_improve,
                             time_contract, time_cutset) )