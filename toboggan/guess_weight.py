#! /usr/bin/env python3
#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# -*- coding: utf-8 -*-
# python libs
import sys
import itertools
# local imports
from toboggan.dp import solve as solve_dp


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1,
                   bar_length=100):
    """
    Call in a loop to create terminal progress bar.

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
                                  complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%',
                     suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def is_feasible(weights, flow, max_weight):
    """Test whether set of guessed weights is feasible."""
    # In the following, we replace very occurenve of 'None' in the
    # weight-array by the minimum/maximum possible value (given by the
    # last/the first
    # non-None value next to it).
    min_weights = [1] + weights
    max_weights = [max_weight] + list(reversed(weights))
    for i in range(1, len(min_weights)):
        min_weights[i] = min_weights[i] if min_weights[i] else min_weights[i-1]
        max_weights[i] = max_weights[i] if max_weights[i] else max_weights[i-1]
    min_weights = min_weights[1:]
    max_weights = list(reversed(max_weights[1:]))

    # If the flow value lies outside of the sum-of-weight estimates,
    # the current guessed set of weights is infeasible.
    return sum(min_weights) <= flow and sum(max_weights) >= flow


def solve(instance, silent=True, max_weight_lower=1,
          max_weight_upper=float('inf')):
    """Solve the provided instance of path-flow decomposition."""
    flow = instance.flow
    k = instance.k

    # quit right away if the instance has weight bounds that can't be satisfied
    if instance.has_bad_bounds():
        return set()

    # if k equals the size of the largest edge cut, the weights are
    # predetermined
    if instance.k == max(len(C) for C in instance.edge_cuts):
        largest_cut = max(instance.edge_cuts, key=len)
        # Important: path weights must be sorted, otherwise our
        # subsequent optimizations will remove this constraint.
        weights = list(sorted(w for _, w in largest_cut))
        return solve_dp(instance, silent=True, guessed_weights=weights)

    max_weight = instance.max_weight_bounds[1]
    feasible_weights = list(filter(lambda w: w <= max_weight,
                                   instance.weights))

    if not silent:
        print(instance.weights, feasible_weights)

    # figure out whether we get some of the first or last positions for free
    largest_free = False
    smallest_free = False
    ones_count = 0
    # check largest weight first
    if instance.max_weight_bounds[0] == instance.max_weight_bounds[1]:
        largest_free = True
        largest = instance.max_weight_bounds[0]
    if min(instance.weights) == 1:
        smallest_free = True
        # check how many times 1 appears as a weight in a cutset
        ones_count = max(sum(1 for e in C if e[1] == 1)
                         for C in instance.edge_cuts)
        smallest = [1 for _ in range(ones_count)]
        # get an early out if an entry that needs to be 1 conflicts with its
        # lower bound
        for upper, lower in instance.weight_bounds[:ones_count]:
            if lower != 1:
                return set()

    positions = list(range(ones_count, k-int(largest_free)))

    # iterate over the number of unguessed weights
    for diff in range(k+1):
        if not silent:
            print("Diff =", diff)
        # iterate over positions of guessed weights.  We want them to be
        # ordered, but choose the smallest first to be removed
        for rev_indices in itertools.combinations(reversed(positions), k-diff):
            indices = list(reversed(rev_indices))
            p = len(indices)
            # when k-1 values are determined, it also determines the kth value
            if p == k-1:
                continue
            # iterate over choices for those guessed weights
            for chosen_weights in itertools.combinations(feasible_weights, p):
                weights = [None] * k

                # assign the chosen weights to the guessed positions
                for p, w in zip(indices, chosen_weights):
                    weights[p] = w

                # add in free values
                if smallest_free:
                    weights[:ones_count] = smallest
                    assert len(weights) == k
                if largest_free:
                    weights[k-1] = largest

                # quit if this didn't work
                if not is_feasible(weights, flow, max_weight):
                    continue

                if not silent:
                    print("Trying weights", weights)
                sol = solve_dp(instance, silent=True, guessed_weights=weights)
                if len(sol) > 0:
                    if not silent:
                        try:
                            for s in sol:
                                print(s, sum(s.path_weights), flow)
                        except AttributeError:
                            print("Unterdetermined solution")
                    return sol
