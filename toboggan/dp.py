#! /usr/bin/env python3
#
# This file is part of Toboggan, https://github.com/TheoryInPractice/Toboggan/,
# and is Copyright (C) North Carolina State University, 2017. It is licensed
# under the three-clause BSD license; see LICENSE.
#
# python libs
import time
from operator import itemgetter
from collections import defaultdict, deque

# local imports
from toboggan.flow import Instance, Constr, SolvedConstr, PathConf


def solve(instance, silent=True, guessed_weights=None):
    dpgraph = instance.dpgraph
    k = instance.k
    n = instance.n

    if not silent:
        print(dpgraph)

    if instance.max_weight_bounds[0] > instance.max_weight_bounds[1]:
        return set()  # Instance is not feasible
    # The only constraint a priori is the total flow value
    globalconstr = Constr(instance)
    guessed_weights = [None]*k if not guessed_weights else guessed_weights

    # We sometimes obtain tight bounds for the highest weight. This either
    # gives us the highest weight for free or contradicts a guessed weight.
    if instance.max_weight_bounds[0] == instance.max_weight_bounds[1]:
        if guessed_weights[-1] is None:
            guessed_weights[-1] = instance.max_weight_bounds[0]
        elif guessed_weights[-1] != instance.max_weight_bounds[0]:
            return set()  # Guessed weights not feasible

    # If there exists a weight-one edge we know what the smallest weight
    # should be.
    if min(instance.weights) == 1:
        if guessed_weights[0] is None:
            guessed_weights[0] = 1
        elif guessed_weights[0] != 1:
            return set()  # Guessed weights not feasible

    assert len(guessed_weights) == k
    for index, guess in enumerate(guessed_weights):
        if guess is None:
            continue
        globalconstr = globalconstr.add_constraint([index], (0, guess))

    # Check whether there is a cut-vertex of out-degree k,
    # this lets us derive the path values directly.
    # Also check whether any active set has size larger than k, in which case
    # this is a no-instance
    for i, cut in enumerate(instance.cuts):
        if len(cut) > k:
            return set()  # Early out: no solution
        if len(cut) == 1 and len(dpgraph[i]) == k:
            # Important: path weights must be sorted, otherwise our
            # subsequent optimizations will remove this constraint.
            weights = sorted(map(itemgetter(1), dpgraph[i]))
            if not silent:
                print("Found bottleneck. Path weights are {}".format(weights))
            globalconstr = SolvedConstr(weights, instance)

    # Build first DP table
    old_table = defaultdict(set)
    allpaths = frozenset(range(k))
    # All k paths `end' at source
    old_table[PathConf.init(0, allpaths)] = set([globalconstr])

    # Run DP
    for i in range(n-1):
        if not silent:
            print("")
            print("Active ({}): {}".format(i, instance.cuts[i]))
            print("Removing {} from active set".format(i))

        new_table = defaultdict(set)
        for paths, constraints in old_table.items():
            # Distribute paths incoming to i onto its neighbours
            if not silent:
                print("  Pushing {}".format(paths))
            for newpaths, dist in paths.push(i, dpgraph[i]):
                # if not silent:
                    # print("  Candidate paths", newpaths)
                debug_counter = 0
                for constr in constraints:
                    # print("  Combining w/ constraints", constraints)
                    # Update constraint-set constr with new constraints imposed
                    # by our choice of pushing paths o..........es (as
                    # described by dist)
                    newconstr = constr
                    try:
                        for e, p in dist:  # Paths p coincide on edge e
                            newconstr = newconstr.add_constraint(p, e)
                    except ValueError as e:
                        print("Problem while adding constraint", p, e)
                        raise e
                    except AttributeError:
                        assert(newconstr is None)
                        pass

                    if newconstr is None:
                        pass  # Infeasible constraints
                    elif newconstr.is_redundant(newpaths):
                        if not silent:
                            print(".", end="")
                            debug_counter += 1
                            if debug_counter > 80:
                                debug_counter = 0
                                print()
                        pass  # Redundant constraints
                    else:
                        if not silent:
                            if newpaths not in new_table or \
                                    newconstr not in new_table[newpaths]:
                                print()
                                print("    New path-constr pair",
                                      newpaths, newconstr)
                        new_table[newpaths].add(newconstr)  # Add to DP table
        old_table = new_table

    if not silent:
        print("\nDone.")
        print(new_table)

    candidates = new_table[PathConf.init(n-1, allpaths)]
    return candidates


def solve_and_recover(instance, weights, silent=True):
    dpgraph = instance.dpgraph
    k = instance.k
    n = instance.n

    if not silent:
        print(dpgraph)

    # since we know all the weights, we can stored them as a solved constraint
    # system
    globalconstr = SolvedConstr(weights, instance)

    assert len(weights) == k
    allpaths = frozenset(range(k))

    # Build DP table, which will map path configurations to the path
    # configurations in the previous tables that yielded the specific entry
    # All k paths start at the source vertex
    initial_entries = {PathConf.init(0, allpaths): None}
    backptrs = [initial_entries]

    # Run DP
    for i in range(n-1):
        if not silent:
            print("")
            print("Active ({}): {}".format(i, instance.cuts[i]))
            print("Removing {} from active set".format(i))

        entries = {}
        for old_paths in backptrs[-1].keys():
            # Distribute paths incoming to i onto its neighbors
            if not silent:
                print("  Pushing {}".format(old_paths))
            for new_paths, dist in old_paths.push(i, dpgraph[i]):
                # make sure the sets of paths that were pushed along each
                # edge sum to the proper flow value.  If not, don't create a
                # new table entry for this path set.
                for e, P in dist:  # Paths P coincide on edge e
                    success = globalconstr.add_constraint(P, e)
                    if success is None:
                        break
                else:
                    entries[new_paths] = old_paths

        # add the new entries to the list of backpointers
        backptrs.append(entries)

    if not silent:
        print("\nDone.")
        print(backptrs[n-1])

    # recover the paths
    full_paths = [deque() for _ in weights]
    try:
        # the "end" of the DP is all paths passing through the sink
        conf = PathConf.init(n-1, allpaths)
        # iterate over the backpointer list in reverse
        for table in reversed(backptrs):
            for v, incidence in conf:
                # vertices might repeat in consecutive table entries if an edge
                # is "long" wrt the topological ordering.  Don't add it twice
                # to the path lists in this case.
                for p in incidence:
                    if len(full_paths[p]) == 0 or full_paths[p][-1] != v:
                        full_paths[p].appendleft(v)
            # traverse the pointer backwards
            conf = table[conf]
    except ValueError as e:
        raise Exception("The set of weights is not a valid solution") from e
    
    return full_paths

