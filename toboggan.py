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
from toboggan.parser import read_instances_verbose
from toboggan.graphs import cut_reconf
from toboggan.flow import Instance

# Timeout context, see
#   http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
class timeout:
    """
    Enables skipping an input graph-instance once our algorithm has run for a specified amount of time without terminating.
    """
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def index_range(raw):
    """
    Helper function to parse index ranges separated by commas.
    e.g. '1,2-5,7-11'
    """
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


"""
    Main script
"""
parser = argparse.ArgumentParser(description='Splice flows into paths')
parser.add_argument('file', help='a .graph file.'
                    ' Needs a .truth file in the same folder.')
parser.add_argument('--timeout', help='Timeout in seconds', type=int)
parser.add_argument('--disprove', help='Run instance with parameter k-1 '
                    'instead of k (needs a .truth file)', action='store_true')
parser.add_argument('--indices',
                    help='Either a file containing indices '
                    '(position in .graph file) on which to run, '
                    'or a list of indices separated by commas. '
                    'Ranges are accepted as well, e.g. "1,2-5,6"',
                    type=index_range)

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
    print("# Using ground-truth from file {}".format(truth_file))
else:
    print("# No ground-truth found. Guessing parameter.".format(truth_file))
    truth_file = None


for graphdata, k, index in read_instances_verbose(graph_file,
                                                          truth_file):
    graph, graphname, graphnumber = graphdata

    if instances and index not in instances:
        continue
    start = time.time()
    reduced = graph.contracted()

    n = len(reduced)
    m = len(list(reduced.edges()))

    if len(reduced) <= 1:
        print("{} {}:{} is trivial.\n".format(index, graphname, graphnumber))
        continue

    if args.disprove and k:
        k = k - 1
        print("# Using parameter k-1")
    print("{} {}:{} with n = {}, m = {}, and k = {}: ".format(
        index, graphname, graphnumber, n, m, k if k else "?"), flush=True)

    # Reduce and reconfigure graph 
    reduced = cut_reconf(reduced)

    if k:
        instance = Instance(reduced, k)
        if maxtime:
            try:
                with timeout(seconds=maxtime):
                    solutions = solve(instance, silent=True)
                    elapsed = time.time() - start
                    print("Computation took {:.2f} seconds".format(elapsed))
                    print("Solutions:", solutions)
            except TimeoutError:
                print("Timed out after {} seconds".format(maxtime))
        else:
            solutions = solve(instance, silent=True)
            elapsed = time.time() - start
            print("Computation took {:.2f} seconds".format(elapsed))
            print("Solutions:", solutions)
    else:
        if maxtime:
            try:
                with timeout(seconds=maxtime):
                    k = 1
                    solutions = None
                    while solutions == None:
                        instance = Instance(reduced, k)
                        solutions = solve(instance, silent=True)
                        k += 1
                    elapsed = time.time() - start
                    print("Computation took {:.2f} seconds".format(elapsed))
                    print("Solutions:", solutions)                        
            except TimeoutError:
                print("Timed out after {} seconds".format(maxtime))
        else:
            k = 1
            solutions = None
            while solutions == None:
                instance = Instance(reduced, k)
                solutions = solve(instance, silent=True)
                k += 1
            elapsed = time.time() - start
            print("Computation took {:.2f} seconds".format(elapsed))
            print("Solutions:", solutions)        
    print("")     
