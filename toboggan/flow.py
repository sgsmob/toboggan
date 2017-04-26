#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# python libs
import math
from collections import defaultdict
import itertools
import numpy as np
from scipy.optimize import linprog

# local imports
from toboggan.graphs import convert_to_top_sorting, compute_cuts,\
                            compute_edge_cuts
from toboggan.partition import algorithm_u


class Instance:
    """
    Information about an input instance to flow decomposition.

    Maintains a topological ordering of the graph, the flow, and bounds on the
    feasible path weights.
    """

    def __init__(self, graph, k=None):
        """Create an instance from a graph and guess for the solution size."""
        # information about the graph and its ordering
        self.graph = graph
        self.ordering = convert_to_top_sorting(graph)
        self.cuts = compute_cuts(self.graph, self.ordering)
        self.edge_cuts = compute_edge_cuts(self.graph, self.ordering)
        self.flow = sum(w for _, w in self.graph.neighborhood(
                                            self.graph.source()))
        # get a lower bound on the number of paths needed
        # We pass the input k to this function so it can inform the user if
        # the input k can be easily identified as too small.
        self.k = self._optimal_size_lower_bound(k)

        # our initial guesses for weights will come from the flow values
        self.weights = sorted(set([w for _, _, w in self.graph.edges()]))

        # compute bounds on the largest weight
        self.max_weight_bounds = self._compute_max_weight_bounds()
        # compute bounds on the individual weights
        self.weight_bounds = self._compute_weight_bounds()

    def info(self):
        """A string representation of this object."""
        print("n = {}, m = {}, k = {}.".format(len(self.graph),
                                               self.graph.num_edges(), self.k))
        print("Weights:", self.weights)
        print("Max-weight bounds:", self.max_weight_bounds)
        print("Weight bounds:", list(map(tuple, self.weight_bounds)))
        print("")
        print("Cut-representation:")
        print(self.cuts)

    def _compute_max_weight_bounds(self):
        # Get lower bound for highest weight
        min_max_weight = 1
        # This checks each topological cut
        for topol_cut in self.edge_cuts:
            cut_size = len(topol_cut)
            for _, w in topol_cut:
                # use pigeonhole principle to lowerbound max weight
                min_max_weight = max(min_max_weight, w // (self.k-cut_size+1))

        # Compute heaviest possible path in graph
        # by iterating over each node's out-neighborhood
        maxpath = {v: 0 for v in self.graph}
        maxpath[self.graph.source()] = self.flow
        for v in self.ordering:
            out = self.graph.neighborhood(v)
            for u, w in out:
                maxpath[u] = max(maxpath[u], min(w, maxpath[v]))

        return (min_max_weight, maxpath[self.graph.sink()])

    def _compute_weight_bounds(self):
        supseq = []
        summed = 0
        # supseq is a list of "super-increasing" values taken from edge weights
        # starting from smallest weight in the graph. These values are upper
        # bounds on the different path weights.
        for w in self.weights:
            if w > self.max_weight_bounds[1]:
                break
            if w > summed:
                supseq.append(w)
                summed += w
        # pad the rest of supseq with the max_weight_bound
        while len(supseq) < self.k:  # Sentinel elements
            supseq.append(self.max_weight_bounds[1])

        bounds = [(1, w) for w in supseq[:self.k]]
        bounds[-1] = self.max_weight_bounds

        # Next, compute lowerbounds for the path weights.
        uppersum = [u for _, u in bounds]
        for i in reversed(range(self.k-1)):
            uppersum[i] += uppersum[i+1]

        # Refine lower bounds by using upper bounds:
        # the weight of path i must be at least F_i / i
        # where F_i is an upper bound on how much flow all paths > i
        # take up.
        for i in range(1, self.k-1):
            lower = max(bounds[i][0], (self.flow-uppersum[i+1]) // (i+1))
            bounds[i] = (lower, bounds[i][1])
        return np.array(bounds)

    def _larger_multiset_diff(self, list1, list2):
        """
        Treat twolists as multisets, return list1-list2.
        Note: input lists should contain int or float type.
        """
        # convert to dicts with contents as keys, multiplicities as vals
        dict1 = defaultdict(int)
        for item in list1:
            dict1[item] += 1
        dict2 = defaultdict(int)
        for item in list2:
            dict2[item] += 1
        diffsum12 = 0
        for key, val in dict1.items():
            temp_diff = val - dict2[key]
            if temp_diff > 0:
                diffsum12 += temp_diff
        diffsum21 = 0
        for key, val in dict2.items():
            temp_diff = val - dict1[key]
            if temp_diff > 0:
                diffsum21 += temp_diff

        return max(diffsum12, diffsum21)

    def _optimal_size_lower_bound(self, k):
        """
        Get a lower bound on the optimal solution size.

        We look over all s-t edge cuts consistent with the topological ordering
        and pick the largest. Then we look over all pairs of cut-sets that are
        large enough to further improve this lower-bound and check whether the
        number of distinct edge-weights requires a larger lower-bound than
        merely the largest cut-set size.
        """
        edge_cut_sizes = [len(C) for C in self.edge_cuts]
        max_edge_cut = max(edge_cut_sizes)
        lower_bound = max_edge_cut

        # Now check all pairs of cutsets "large enough" for better bound
        sorted_cut_sizes = sorted([(cut_size, which_cut) for which_cut,
                                   cut_size in enumerate(edge_cut_sizes)],
                                  reverse=True)
        cutsets_for_best_bound = []
        # Starting with largest, iterate over cutsets
        for idx1 in range(len(sorted_cut_sizes)):
            current_size1, which_cut1 = sorted_cut_sizes[idx1]
            # once one set is too small, all following will be, so break out
            if math.ceil(current_size1/2) + current_size1 <= lower_bound:
                break
            for idx2 in range(idx1+1, len(sorted_cut_sizes)):
                current_size2, which_cut2 = sorted_cut_sizes[idx2]
                # if cutsize2 too small, the rest will be: break inner for loop
                temp_bound = min(current_size1, current_size2) + math.ceil(
                    max(current_size1, current_size2)/2)
                if temp_bound <= lower_bound:
                    break
                # Now compute actual bound for this pair of cutsets;
                # Get weights for each cutset as a multiset,
                # compute size of (larger) difference
                weights1 = set([w for _, w in self.edge_cuts[which_cut1]])
                weights2 = set([w for _, w in self.edge_cuts[which_cut2]])
                multiset_diff = self._larger_multiset_diff(weights1, weights2)
                bound = math.ceil(multiset_diff/2) + min(current_size1,
                                                         current_size2)
                # Check if we need to update bound
                if bound > lower_bound:
                    lower_bound = bound
        # let the user know their guess was bad if it was
        print("#\tGraph has an edge cut of size {}.\n"
              "#\tInvestigating cutsets yields bound {}.\n"
              "#\tUser supplied k value of {}.\n"
              "#\tContinuing using k = {}"
              "".format(max_edge_cut, lower_bound, k, lower_bound))
        if k is not None and lower_bound > k:
            return lower_bound
        elif k is None:
            return lower_bound
        else:
            return k

    def try_larger_k(self):
        """
        Increase the value of k by 1.

        We need to do this in a method in order to update internal data
        structures about the weights.
        """
        self.k = self.k + 1
        # compute bounds on the largest weight
        self.max_weight_bounds = self._compute_max_weight_bounds()
        # compute bounds on the individual weights
        self.weight_bounds = self._compute_weight_bounds()

    def has_bad_bounds(self):
        """Check whether weight bounds disallow all solutions."""
        # upper weight bounds miss each other
        if self.max_weight_bounds[0] > self.max_weight_bounds[1]:
            return True
        # lower and upper bounds of each weight position miss each other
        for lower, upper in self.weight_bounds:
            if lower > upper:
                return True
        # otherwise all good
        return False


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

    def is_redundant(self):
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

        # wrap linear program in try block to check when we have bounds that
        # are invalid.
        try:
            optres = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
            if optres.status == 2:
                # System is infeasible and therefore redundant.
                return True
        except ValueError as e:
            # bounds that don't make sense indicate an infeasible system
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

    def is_redundant(self):
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

    def __init__(self, vertex=None, paths=None):
        self.paths = {}
        if vertex is not None:
            self.paths[vertex] = frozenset(paths)

    def copy(self):
        res = PathConf()
        for v, paths in self.paths.items():
            res.paths[v] = frozenset(paths)
        return res

    def __iter__(self):
        # return iter(self.paths.items())
        # THIS CHANGE POTENTIALLY DANGEROUS
        # so intead we didn't make this change,
        #  and altered the way dp.solve_and_recover iterates over conf
        return iter(self.paths)

    def __contains__(self, v):
        return v in self.paths

    def __getitem__(self, v):
        return self.paths[v]

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        res = "PathConf("
        for v in self.paths:
            res += "{}: {}, ".format(v, list(self.paths[v]))
        return res[:-2] + ")"

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
        if len(self.paths[v]) == 0:
            raise ValueError("{} has no paths running through it.".format(v))
        if len(edges) == 0:
            raise ValueError("{} has no edges exiting it".format(v))
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
