#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# local imports
import toboggan.graphs
from toboggan.graphs import AdjList
# python libs
import os
import re

header_regex = re.compile('# graph number = ([0-9]*) name = (.*)')
edge_regex = re.compile('(\d*) (\d*) (\d*\.\d*)')


def enumerate_graphs(graph_file):
    for x in enumerate_graphs_verbose(graph_file):
        yield x[0]


def enumerate_graphs_verbose(graph_file):
    def read_next_graph(f):
        header_line = f.readline()

        if header_line == '':
            return None

        m = header_regex.match(header_line)
        if m is None:
            raise Exception('Misformed graph header line.')
        (graph_number, graph_name) = (m.group(1), m.group(2))

        line = f.readline()
        num_nodes = int(line.strip())

        graph = AdjList(graph_file, graph_number, graph_name, num_nodes)

        while not line == '':
            last_pos = f.tell()
            line = f.readline()

            if line == '':
                break
            elif line[0] == '#':
                f.seek(last_pos)
                break

            list = line.split()

            u = int(list[0])
            v = int(list[1])
            flow = int(float(list[2]))

            graph.add_edge(u, v, flow)

        return graph, graph_name, graph_number

    with open(graph_file) as f:
        while True:
            graph_data = read_next_graph(f)
            if graph_data is None:
                break
            else:
                yield graph_data


def enumerate_decompositions(decomposition_file):
    for x in enumerate_decompositions_verbose(decomposition_file):
        yield (x[0], x[2])  # Return name & decomposition


def enumerate_decompositions_verbose(decomposition_file):
    def read_next_decomposition(f):
        header_line = f.readline()

        if header_line == '':
            return None

        m = header_regex.match(header_line)
        if m is None:
            raise Exception('Misformed graph header line.')
        (graph_number, graph_name) = (m.group(1), m.group(2))

        path_decomposition = []
        line = header_line
        while not line == '':
            last_pos = f.tell()
            line = f.readline()

            if line == '':
                break
            elif line[0] == '#':
                f.seek(last_pos)
                break

            l = line.split()
            l = list(map(lambda x: int(x), l))

            path_decomposition.append((l[0], l[1:]))

        return (graph_name, graph_number, path_decomposition)

    with open(decomposition_file) as f:
        while True:
            decomposition = read_next_decomposition(f)
            if decomposition is None:
                break
            else:
                yield decomposition


def append_graph_to_file(graph, filename):
    with open(filename, "a") as f:
        f.write("# graph number = {} name = {}\n".format(graph.graph_number,
                graph.name))
        f.write(str(graph.num_nodes()) + "\n")
        for (u, v, flow) in graph.edges():
            f.write("{} {} {}\n".format(u, v, flow))


def read_instances(graph_file, truth_file, indices=None):
    if indices:
        indices_set = set(indices)
        max_index = max(indices)

        index = 0
        for (graph, truth) in zip(enumerate_graphs(graph_file),
                                  enumerate_decompositions(truth_file)):
            index += 1
            if index in indices_set:
                yield (graph, truth, index)

            if index > max_index:
                break
    else:
        index = 0
        for (graph, truth) in zip(enumerate_graphs(graph_file),
                                  enumerate_decompositions(truth_file)):
            index += 1
            yield (graph, truth, index)


def read_instances_verbose(graph_file, truth_file):
    index = 0
    if truth_file:
        for graphdata, truthdata in zip(enumerate_graphs_verbose(graph_file),
                                        enumerate_decompositions_verbose(
                                        truth_file)):
            index += 1
            _, _, solution = truthdata
            yield (graphdata, len(solution), index)
    else:
        for graphdata in enumerate_graphs_verbose(graph_file):
            index += 1
            yield (graphdata, None, index)