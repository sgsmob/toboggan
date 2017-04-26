#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
import copy
from collections import defaultdict
from operator import itemgetter
import itertools


class AdjList:
    def __init__(self, graph_file, graph_number, name, num_nodes):
        self.graph_file = graph_file
        self.graph_number = graph_number
        self.name = name
        self.num_nodes_at_start = num_nodes
        self.vertices = set()
        self.adj_list = defaultdict(list)
        self.inverse_adj_list = defaultdict(list)

    def subgraph(self, vertices):
        res = AdjList(self.graph_file, self.graph_number, self.name,
                      len(vertices))
        for x in vertices:
            for y, f in self.out_neighborhood(x):
                if y in vertices:
                    res.add_edge(x, y, f)
        return res

    def append(self, other):
        """Identify sink of this graph with source of the next graph."""
        sink = self.sink()
        other_source = other.source()

        # Make sure vertex ids are disjoint
        vertices = set(list(iter(self)))
        index = max(vertices) + 1
        remap = {}
        for v in other:
            if v in vertices:
                remap[v] = index
                index += 1
            else:
                remap[v] = v

        # Identify source of other graph with this one's sink
        remap[other_source] = sink

        for u, w, f in other.edges():
            self.add_edge(remap[u], remap[w], f)

    def add_edge(self, u, v, flow):
        self.vertices.add(u)
        self.vertices.add(v)
        self.adj_list[u].append((v, flow))
        self.inverse_adj_list[v].append((u, flow))

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def source(self):
        for v in self:
            if self.in_degree(v) == 0:
                return v
        raise TypeError("This graph has no source")

    def sink(self):
        for v in self:
            if self.out_degree(v) == 0:
                return v
        raise TypeError("This graph has no sink")

    def neighborhood(self, u):
        if u in self.adj_list:
            return self.adj_list[u]
        else:
            return []

    def out_neighborhood(self, u):
        return self.neighborhood(u)

    def in_neighborhood(self, u):
        if u in self.inverse_adj_list:
            return self.inverse_adj_list[u]
        else:
            return []

    def remove_weight(self, weight, u, v):
        for (w, flow) in self.adj_list[u]:
            if w == v:
                self.adj_list[u].remove((w, flow))
                self.adj_list[u].append((w, flow - weight))
                self.inverse_adj_list[v].remove((u, flow))
                self.inverse_adj_list[v].append((u, flow - weight))
                break

    def copy(self):
        res = AdjList(self.graph_file, self.graph_number, self.name,
                      len(self))
        res.adj_list = copy.deepcopy(self.adj_list)
        res.inverse_adj_list = copy.deepcopy(self.inverse_adj_list)
        res.vertices = set(self.vertices)
        return res

    def edges(self):
        for u in self.adj_list:
            for (v, flow) in self.adj_list[u]:
                yield (u, v, flow)

    def num_edges(self):
        return sum(1 for _ in self.edges())

    def reverse_edges(self):
        for v in self.inverse_adj_list:
            for (u, flow) in self.inverse_adj_list[v]:
                yield (u, v, flow)

    def out_degree(self, v):
        return len(self.out_neighborhood(v))

    def in_degree(self, v):
        return len(self.in_neighborhood(v))

    def contracted(self):
        """
        Return a copy of the graph in which all uv arcs where u has out degree
        1 or v has in degree 1 are contracted.
        """
        res = self.copy()
        arc_mapping = {e: [e] for e in res.arcs()}
        # contract out degree 1 vertices
        for u in list(res):
            if res.out_degree(u) == 1:
                arc = res.out_arcs(u)[0]
                # mark u's inarcs to know they use the arc to be contracted
                for a in res.in_arcs(u):
                    arc_mapping[a].extend(arc_mapping[arc])
                # contract the edge
                res.contract_edge(a, keep_source=False)
        # contract in degree 1 vertices
        for v in list(res):
            if res.in_degree(v) == 1:
                arc = res.in_arcs(v)[0]
                # mark v's outarcs to know they use the arc to be contracted
                for a in res.out_arcs(v):
                    new_path = list(arc_mapping[arc])
                    arc_mapping[a] = new_path.extend(arc_mapping[a])
                # print("{} has in degree 1 from {}".format(v,u))
                res.contract_edge(a, keep_source=True)
        return res, arc_mapping

    def contract_edge(self, e, keep_source=True):
        """
        Contract the arc e.

        If keep_source is true, the resulting vertex retains the label of the
        source, otherwise it keeps the sink's
        """
        u, v, w = self.arc_info[e]
        i = self.out_arcs[u].find(e)
        j = self.in_arcs[v].find(e)
        # move last neighbor into position of uv arc and delete arc
        self.adj_list[u][i] = self.adj_list[u][-1]
        self.adj_list[u] = self.adj_list[u][:-1]
        self.out_arcs[u][i] = self.out_arcs[u][-1]
        self.out_arcs[u] = self.out_arc[u][:-1]

        # move last neighbor into position of uv arc and delete arc
        self.inverse_adj_list[v][j] = self.inverse_adj_list[v][-1]
        self.inverse_adj_list[v] = self.inverse_adj_list[v][:-1]
        self.in_arcs[u][i] = self.in_arcs[u][-1]
        self.in_arcs[u] = self.in_arcs[u][:-1]

        # to keep things concise, use the label a for the vertex to keep
        # and label b for the vertex to discard
        a, b = (u, v) if keep_source else (v, u)

        # update out-neighbors of a
        self.adj_list[a].extend(self.out_neighborhood(b))
        self.out_arcs[a].extend(self.out_arcs[b])
        # make out-neighbors of b point back to a
        for lab, edge in zip(self.out_arcs[b], self.out_neighborhood(b)):
            w, f = edge
            i = self.inverse_adj_list[w].index((b, f))
            self.arc_list[lab] = (a, w, f)
            self.inverse_adj_list[w][i] = (a, f)

        # update in-neighbors of a
        self.inverse_adj_list[a].extend(self.in_neighborhood(b))
        self.in_arcs[a].extend(self.in_arcs[b])
        # make in neighbors of b point to a
        for lab, edge in self.in_neighborhood(b):
            w, f = edge
            i = self.adj_list[w].index((b, f))
            self.arc_list[lab] = (w, a, f)
            self.adj_list[w][i] = (a, f)

        if b in self.adj_list:
            del self.adj_list[b]
        if b in self.inverse_adj_list:
            del self.inverse_adj_list[b]
        self.vertices.remove(b)
        del self.arcs_info[e]

    def reversed(self):
        res = AdjList(self.graph_file, self.graph_number, self.name,
                      self.num_nodes())
        for u, v, w in self.edges():
            res.add_edge(v, u, w)
        return res

    def show(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        for u, w, f in self.edges():
            G.add_edge(u, w)
        nx.draw(G)
        plt.show()


def test_solution(graph, solution):
    arc_weights = defaultdict(int)
    vertices = set()
    for weight, path in solution[1]:
        for u, v in zip(path[:-1], path[1:]):
            arc_weights[(u, v)] += weight
            vertices.add(u)
            vertices.add(v)

    if set(graph) != vertices:
        print("Vertex sets are different:")
        print("  Graph has vertices", set(graph))
        print("  Solution has vertices", vertices)
        return
    else:
        print("Graph and solution have same vertex set.")

    arc_count = 0
    for u, v, w in graph.edges():
        if arc_weights[(u, v)] != w:
            print("Arc ({},{}) has weight {} in the solution but {} in the"
                  "graph.".format(u, v, arc_weights[(u, v)], w))
        arc_count += 1
    if arc_count != len(arc_weights):
        print("Number of arcs in solution is different than number of arcs in"
              "graph.")
    else:
        print("Solution and graph produce the same number of arcs.")


def convert_to_top_sorting(graph):
    # 1 temporary marked, 2 is finished
    source = graph.source()
    marks = {}

    def visit(n, ordering):
        if n in marks and marks[n] == 1:
            raise Exception('This graph is not a DAG: ' + graph.name)
        if n not in marks:
            marks[n] = 1
            for (m, _) in graph.neighborhood(n):
                visit(m, ordering)
            marks[n] = 2
            ordering.insert(0, n)

    ordering = []
    visit(source, ordering)

    return ordering


def top_sorting_graph_representation(graph, ordering):
    n = len(ordering)
    mapping = {}

    res = []

    for i, v in enumerate(ordering):
        mapping[v] = i

    for i in range(n):
        u = ordering[i]
        neighborhood = graph.neighborhood(u)
        dag_neighborhood = list(map(lambda pair: (mapping[pair[0]], pair[1]),
                                neighborhood))
        res.append(dag_neighborhood)

    return res


def compute_cuts(graph, ordering):
    """Compute the topological vertex cuts."""
    cuts = [None for v in graph]
    cuts[0] = set([graph.source()])

    for i, v in enumerate(ordering[:-1]):
        # Remove i from active set, add neighbors
        cuts[i+1] = set(cuts[i])
        cuts[i+1].remove(v)
        for t, w in graph.out_neighborhood(v):
            cuts[i+1].add(t)
    return cuts


def compute_edge_cuts(graph, ordering):
    """Compute the topological edge cuts."""
    # Contains endpoints and weights for arcs in each topological cut
    top_cuts = []
    # Contains the iteratively constructed top-edge-cut.
    # key is a node; value is a list of weights of arcs ending at key
    current_bin = defaultdict(list)

    # iterate over nodes in top ordering
    for v in ordering:
        v_neighborhood = graph.neighborhood(v)
        # remove from iterative cut-set the arcs ending at current node
        current_bin[v] = []
        for u, w in v_neighborhood:
            current_bin[u].append(w)
        # current cut-set done, add it to the output
        eC = []
        for u, weights in current_bin.items():
            eC.extend((u, weight) for weight in weights)
        top_cuts.append(eC)

    return top_cuts
