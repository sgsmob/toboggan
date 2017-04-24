#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
import copy
from collections import defaultdict
from operator import itemgetter


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
        # contract out degree 1 vertices
        for u in list(res):
            if res.out_degree(u) == 1:
                v = res.out_neighborhood(u)[0][0]
                # print("{} has out degree 1 to {}".format(u,v))
                res.contract_edge(u, v, keep_source=False)
        # contract in degree 1 vertices
        for v in list(res):
            if res.in_degree(v) == 1:
                u = res.in_neighborhood(v)[0][0]
                # print("{} has in degree 1 from {}".format(v,u))
                res.contract_edge(u, v, keep_source=True)
        return res

    def contract_edge(self, u, v, keep_source=True):
        """
        Contract the arc from u to v.

        If keep_source is true, the resulting vertex retains the label of the
        source, otherwise it keeps the sink's
        """
        # print("Edges:")
        # for e in sorted(self.edges()):
        #    print(e)
        # print("Inverse Edges:")
        # for e in sorted(self.reverse_edges()):
        #    print(e)
        # delete the arc u v
        # find where v is in the adjacency list
        for i, edge in enumerate(self.out_neighborhood(u)):
            w, f = edge
            if w == v:
                break
        else:
            raise ValueError("{} not an out neighbor of {}".format(v, u))
        # find where u is in the inverse adjacency list
        for j, edge in enumerate(self.in_neighborhood(v)):
            w, f = edge
            if w == u:
                break
        else:
            raise ValueError("{} not an in neighbor of {}".format(u, v))
        # move last neighbor into position of uv arc and delete arc
        self.adj_list[u][i] = self.adj_list[u][-1]
        self.adj_list[u] = self.adj_list[u][:-1]

        # move last neighbor into position of uv arc and delete arc
        self.inverse_adj_list[v][j] = self.inverse_adj_list[v][-1]
        self.inverse_adj_list[v] = self.inverse_adj_list[v][:-1]

        # to keep things concise, use the label a for the vertex to keep
        # and label b for the vertex to discard
        a, b = (u, v) if keep_source else (v, u)

        # update out-neighbors of a
        self.adj_list[a].extend(self.out_neighborhood(b))
        # make out-neighbors of b point back to a
        for w, f in self.out_neighborhood(b):
            i = self.inverse_adj_list[w].index((b, f))
            self.inverse_adj_list[w][i] = (a, f)

        # update in-neighbors of a
        self.inverse_adj_list[a].extend(self.in_neighborhood(b))
        # make in neighbors of b point to a
        for w, f in self.in_neighborhood(b):
            i = self.adj_list[w].index((b, f))
            self.adj_list[w][i] = (a, f)

        if b in self.adj_list:
            del self.adj_list[b]
        if b in self.inverse_adj_list:
            del self.inverse_adj_list[b]
        self.vertices.remove(b)

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


class DiGraph:
    def __init__(self):
        self.vertices = set()
        self.out_arcs = defaultdict(set)
        self.in_arcs = defaultdict(set)
        self.labels = defaultdict(list)
        self.arc_count = 0

    def add_arc(self, u, v, label):
        self.vertices.add(u)
        self.vertices.add(v)
        self.out_arcs[u].add(v)
        self.in_arcs[v].add(u)

        if len(self.labels[(u, v)]) == 0:
            self.arc_count += 1
        self.labels[(u, v)].append(label)

    def num_labels(self, u, v):
        return len(self.labels[(u, v)])

    def arc_labels(self, u, v):
        return self.labels[(u, v)]

    def remove_arc(self, u, v):
        self.in_arcs[v].remove(u)
        self.out_arcs[u].remove(v)
        del self.labels[(u, v)]
        self.arc_count -= 1

    def remove_vertex(self, u):
        self.vertices.remove(u)

        for v in self.out_arcs[u]:
            self.in_arcs[v].remove(u)
            del self.labels[(u, v)]
            self.arc_count -= 1
        del self.out_arcs[u]

        for v in self.in_arcs[u]:
            self.out_arcs[v].remove(u)
            del self.labels[(v, u)]
            self.arc_count -= 1
        del self.in_arcs[u]

    def in_degree(self, u):
        return len(self.in_arcs[u])

    def in_neighbours(self, u):
        for v in self.in_arcs[u]:
            yield v

    def out_degree(self, u):
        return len(self.out_arcs[u])

    def out_neighbours(self, u):
        for v in self.out_arcs[u]:
            yield v

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def num_arcs(self):
        return self.arc_count


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


def compute_cuts(dpgraph):
    n = len(dpgraph)
    cuts = [None] * (n)
    cuts[0] = set([0])

    for i in range(n-1):
        # Remove i from active set, add neighbors
        cuts[i+1] = set(cuts[i])
        cuts[i+1].remove(i)
        for t, w in dpgraph[i]:
            cuts[i+1].add(t)
    return cuts


def cut_reconf(graph):
    """
    Decomposes the graph into blocks and re-orders them, with the blocks
    that contain the largest topological separator coming first.
    """
    top_order = convert_to_top_sorting(graph)

    # Built up separators
    n = len(top_order)
    sep = [None] * (n+1)
    sep[0] = set([top_order[0]])

    for i, x in enumerate(top_order):
        # Remove i from active set, add neighbors
        sep[i+1] = set(sep[i])
        sep[i+1].remove(x)
        for y, _ in graph.out_neighborhood(x):
            sep[i+1].add(y)

    sep = sep[:-1]  # Last element is guard element

    # Find cut-vertices and collect blocks
    cutvs = []
    sepsizes = [1]
    blocks = [set()]
    for x, s in zip(top_order, sep):
        cutindex = len(cutvs)
        cutsize = len(s)
        blocks[cutindex] |= s
        if cutsize == 1:
            cutvs.append(x)
            sepsizes.append(1)
            blocks.append(set([x]))
        else:
            sepsizes[cutindex] = max(sepsizes[cutindex], cutsize)
    sepsizes = sepsizes[1:-1]
    blocks = blocks[1:-1]

    # Collate start-vertex, end-vertex, and separator size into single list
    pieces = [(s, t, sepsizes[i], blocks[i]) for i, (s, t) in
              enumerate(zip(cutvs[:-1], cutvs[1:]))]
    pieces = sorted(pieces, key=itemgetter(2), reverse=True)

    subgraphs = []
    for _, _, _, vertices in pieces:
        subgraphs.append(graph.subgraph(vertices))

    res = subgraphs.pop(0)
    for subg in subgraphs:
        res.append(subg)
    return res
