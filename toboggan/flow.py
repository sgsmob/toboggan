#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
from toboggan.graphs import convert_to_top_sorting, top_sorting_graph_representation
from toboggan.partition import algorithm_u
from operator import itemgetter
import itertools
import numpy as np
from scipy.optimize import linprog

class Instance:
    def __init__(self, graph, k):
        self.graph = graph
        self.k = k

        self.ordering = convert_to_top_sorting(graph)
        self.dpgraph = top_sorting_graph_representation(graph, self.ordering)

        self.n = len(self.dpgraph)
        self.flow = sum(map(itemgetter(1), self.dpgraph[0]))
        self.weights = sorted(set([w for _, _, w in self.graph.edges()]))

        self.max_weight_bounds = Instance.compute_max_weight_bounds(
                                 self.dpgraph, self.k, self.flow)
        self.weight_bounds = Instance.compute_weight_bounds(
                             self.max_weight_bounds, self.weights, self.k,
                             self.flow)

        self.cuts = Instance.compute_cuts(self.dpgraph)

    def info(self):
        print("n = {}, m = {}, k = {}.".format(len(self.graph),
                                               self.graph.num_edges(), self.k))
        print("Weights:", self.weights)
        print("Max-weight bounds:", self.max_weight_bounds)
        print("Weight bounds:", list(map(tuple, self.weight_bounds)))
        print("")
        print("Cut-representation:")
        print(self.cuts)

    @staticmethod
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

    @staticmethod
    def compute_max_weight_bounds(dpgraph, k, flow):
        # Get lower bound for highest weight
        min_max_weight = 1
        for out in dpgraph:
            degree = len(out)
            for _, w in out:
                if k-degree+1 <= 0:
                    continue  # Instance is infeasible, will be caught later
                min_max_weight = max(min_max_weight, w // (k-degree+1))

        # Compute heaviest possible path in graph
        maxpath = [0] * len(dpgraph)
        maxpath[0] = flow
        for v, out in enumerate(dpgraph):
            for u, w in out:
                maxpath[u] = max(maxpath[u], min(w, maxpath[v]))

        return (min_max_weight, maxpath[-1])

    @staticmethod
    def compute_weight_bounds(max_weight_bounds, weights, k, flow):
        supseq = []
        summed = 0
        for w in weights:
            if w > max_weight_bounds[1]:
                break
            if w > summed:
                supseq.append(w)
                summed += w
        while len(supseq) < k:  # Sentinel elements
            supseq.append(max_weight_bounds[1])

        bounds = [(1, w) for w in supseq[:k]]
        bounds[-1] = max_weight_bounds

        uppersum = [u for _, u in bounds]
        for i in reversed(range(k-1)):
            uppersum[i] += uppersum[i+1]

        # Refine lower bounds by using upper bounds:
        # the weight of path i must be at least F_i / i
        # where F_i is an upper bound on how much flow all paths > i
        # take up.
        for i in range(1, k-1):
            lower = max(bounds[i][0], (flow-uppersum[i+1]) // i)
            bounds[i] = (lower, bounds[i][1])

        return np.array(bounds)


class Constr:
    """
        Class representing linear constraints imposed on path
        weights as collected by the DP routine.
    """
    eps = np.finfo(float).eps
    ORDER_MATRIX = {}  # Pre-computed matrices with ones on the diagonal and
    #                    -1 on the upper off-diagonal.
    ZERO_VECS = {}     # Pre-computed zero vectors
    INFEASIBLE = 0
    REDUNDANT = 1
    VALID = 2
    SOLVED = 3
    POW2 = None

    def __init__(self, instance):
        self.instance = instance
        self.known_values = [None] * self.instance.k

        # Make sure the necessary constants exist
        if self.instance.k not in Constr.ORDER_MATRIX:
            t = self.instance.k
            Constr.ORDER_MATRIX[t] = np.eye(t-1, t, dtype=int) - \
                np.eye(t-1, t, k=1, dtype=int)
            Constr.ZERO_VECS[t-1] = np.zeros(t-1, dtype=int)
            Constr.POW2 = 2**np.arange(64, dtype=np.uint64)

        row = np.array([1] * self.instance.k + [self.instance.flow])
        # In our application instance.k and instance.flow should always be the
        # same, but we want to keep things clean.
        self.hashvalue = hash(row.data.tobytes()) ^ hash((self.instance.k,
                                                          self.instance.flow))
        self.A = np.matrix(row)

    def __repr__(self):
        return str(self.A)

    def is_redundant(self, paths):
        # We can reduce the number of redundant solutions by imposing that
        # the already known path-weights are sorted.
        weights = [w for w in self.known_values if w is not None]
        if not all(weights[i] <= weights[i+1] for i in range(len(weights)-1)):
            return True

        # The following LP tells us whether there is a feasible (rational)
        # solution whose weights are sorted in ascending order. If that is
        # not the case this set of constraints is redundant.
        t = self.instance.k
        c = np.array([1]*t)  # Optimization not important.
        # Equality constraints: flow values
        A_eq = self.A[:, :-1]
        b_eq = self.A[:, -1]
        # Inequality constraints: ensure that weights are sorted
        A_ub = Constr.ORDER_MATRIX[t]
        assert(A_ub.shape == (t-1, t))
        b_ub = Constr.ZERO_VECS[t-1]
        # Bounds for flow values: at least 1, at most F-(t-1) (since non-zero)
        bounds = self.instance.weight_bounds
        optres = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)

        if optres.status == 2:
            # System is infeasible and therefore redundant.
            return True

        return False

    def _test_row(self, row):
        # Check whether b is lin. dep. on rows in M
        coeff, residuals, rank, _ = np.linalg.lstsq(self.A.transpose(),
                                                    row.transpose())

        if rank == self.instance.k:
            M = self.A[:, :-1]
            b = np.squeeze(np.array(self.A[:, -1]))
            try:
                weights = np.linalg.solve(M, b).T
            except np.linalg.linalg.LinAlgError as e:
                # Matrix is singular: we added a constraint that is linearly
                # dependent to earlier constraints BUT its flow-value is
                # different (otherwise it would be a redundant constraint).
                # We can safely discard this solution.
                return Constr.INFEASIBLE, None
            except ValueError as e:
                print("----------------------------")
                print("  Numpy linalg solver crashed on th following input:")
                print(M)
                print(b)
                print("----------------------------")
                raise e

            # We tried doing the following inside numpy, but ran into a numpy
            # bug
            weights = weights.astype(float).tolist()
            for i, w in enumerate(weights):
                if w <= 0 or not w.is_integer():
                    return Constr.INFEASIBLE, None
                weights[i] = int(w)

            return Constr.SOLVED, SolvedConstr(weights, self.instance)

        if len(residuals) == 0:
            return Constr.INFEASIBLE, None

        # residuals should be positive
        residuals = residuals[0]
        assert residuals >= 0, \
            "Residuals not positive: {} {} {}".format(self.A, row, residuals)

        if residuals < Constr.eps:
            return Constr.REDUNDANT, None

        return Constr.VALID, None

    def add_constraint(self, paths, edge):
        # Convert to constraint row
        flow = edge[1]
        row = np.array([0] * self.instance.k + [flow])
        for i in paths:
            row[i] = 1

        # Early-out: if the flow-value of this edge is smaller
        # than the number of paths across is, no integral solution
        # can exist.
        if len(paths) > flow:
            return None

        row_res, solution = self._test_row(row)
        if row_res == Constr.REDUNDANT:
            return self
        elif row_res == Constr.INFEASIBLE:
            return None
        elif row_res == Constr.SOLVED:
            return solution  # Instance of SolvedConstr

        assert(row_res == Constr.VALID)

        # Find index to insert that maintains order
        i = 0
        fc = self.A.shape[1] - 1  # Column with f-values
        while flow > self.A[i, fc]:
            i += 1

        # Tie-break in case flow values are the same
        # row_repr = np.packbits(row[:-1])
        bitlen = len(row)-1
        # Similar to np.packbits, but works for more than 8 paths.
        row_repr = row[:-1].dot(Constr.POW2[:bitlen])
        try:
            while flow == self.A[i, fc] and \
                    row_repr < self.A[i, :-1].dot(Constr.POW2[:bitlen]):
                i += 1
        except ValueError as e:
            print(">>>>>>>>>>>>>")
            print(row)
            print(row_repr)
            print(self.A[i, :-1])
            print(Constr.POW2[:bitlen])
            print("<<<<<<<<<<<<<")
            raise(e)

        res = Constr(self.instance)
        res.A = np.insert(self.A, i, row, axis=0)
        # update hashvalue by new row
        res.hashvalue ^= hash(row.data.tobytes())
        res.known_values = list(self.known_values)

        # Keep track of path-weights that are determined already.
        if len(paths) == 1:
            res.known_values[paths[0]] = flow

        return res

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        # Since we keep A sorted the following comparisons work.
        if not np.array_equal(self.A, other.A):
            return False

        # This should always be true in our application, added
        # only for completeness.
        if self.instance != other.instance:
            return False

        return True

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return self.hashvalue


class SolvedConstr:
    """
    Special case of a constraint in which all path-weights have
    already been determined.
    """
    def __init__(self, path_weights, instance):
        self.instance = instance
        self.path_weights = tuple(path_weights)

        self.hashvalue = hash(self.path_weights) ^ hash((self.instance.k,
                                                         self.instance.flow))

    def __repr__(self):
        return "SolvedConstr " + str(self.path_weights)

    def is_redundant(self, pathconf):
        # We can reduce the number of redundant solutions by imposing that
        # the path-weights are sorted.
        if not all(self.path_weights[i] <= self.path_weights[i+1]
                   for i in range(self.instance.k-1)):
            return True

        # Also ensure that the guess/bound on the max weight is honored
        if self.path_weights[-1] < self.instance.max_weight_bounds[0] or \
                self.path_weights[-1] > self.instance.max_weight_bounds[1]:
            return True
        return False

    def _test_row(self, row):
        # Check whether row is compatible with path weights
        flow = row[-1]
        summed = np.dot(row[:-1], self.path_weights)

        if flow != summed:
            return Constr.INFEASIBLE, None

        return Constr.SOLVED, self

    def add_constraint(self, paths, edge):
        # Test whether constraint is compatible
        summed = 0
        for i in paths:
            summed += self.path_weights[i]
        if summed == edge[1]:
            return self
        else:
            return None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.instance != other.instance:
            return False

        return self.path_weights == other.path_weights

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return self.hashvalue


class PathConf:
    """Class representing paths ending in a set of vertices."""
    def __init__(self):
        self.paths = {}

    @staticmethod
    def init(vertex, paths):
        res = PathConf()
        res.paths[vertex] = frozenset(paths)
        return res

    def copy(self):
        res = PathConf()
        for v, paths in self.paths.items():
            res.paths[v] = frozenset(paths)
        return res

    def __iter__(self):
        return iter(self.paths)

    def __contains__(self, v):
        return v in self.paths

    def __getitem__(self, v):
        return self.paths[v]

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        res = ""
        for v in self.paths:
            res += "{}: {}, ".format(v, list(self.paths[v]))
        return res[:-2]

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self.paths) != len(other.paths):
            return False
        for v in self.paths:
            if v not in other.paths:
                return False
            if self.paths[v] != other.paths[v]:
                return False
        return True

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self):
        return hash(frozenset(self.paths.items()))

    def push(self, v, edges):
        """
        Return all ways that the paths ending in v can be 'pushed' along the
        provided edges.
        """
        if len(self.paths[v]) < len(edges):
            # There are fewer paths ending in v than out-arcs.
            # This cannot be extended to a solution.
            return

        for dist in distribute(self.paths[v], edges):
            res = self.copy()
            del res.paths[v]  # Copy old paths, remove paths ending in v
            for e, p in dist:  # Push paths over prescribed edges
                t, w = e
                if t in res.paths:
                    res.paths[t] = frozenset(res.paths[t] | set(p))
                else:
                    res.paths[t] = frozenset(p)

            yield res, dist


def distribute(paths, edges):
    """
    Distribute n paths on k edges such that every edge has at least one path.
    """
    k = len(edges)
    paths = list(paths)
    for partition in algorithm_u(paths, k):
        # Generate non-empty partitions with k blocks
        for perm in itertools.permutations(partition):  # Permute to distribute
            yield list(zip(edges, perm))
