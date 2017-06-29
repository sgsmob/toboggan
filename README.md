# Toboggan

Toboggan is a research tool for decomposing a flow on a directed acyclic graph into a minimal number of paths, a problem that commonly occurs in transcript and metagenomic assembly.

## Command line usage
```
$>python3 toboggan.py [INPUT FILE]
```
Runs the Toboggan algorithm on every instance provided in the input file (for file format specifications,
see below or consider the example file provided in `testdata`).

Positional arguments:
  * `file` - a .graph file (see below)

If a `.truth` accompanies the input `.graph` file, the instances are each run with the number of
paths in their respective ground-truth as input. Otherwise, successively larger numbers are
tried until a solution is found.

Optional arguments:
  * `-h, --help` - Show a help message and exit
  * `--timeout` TIMEOUT - Set a timeout for the computation of each individual instance.
  * `--indices` INDICES -  Either a file containing indices (position in `.graph`
                     file) on which to run, or a list of indices separated by
                     commas. Ranges are accepted as well, e.g. "1,2-5,6".

### Scripts for high-throughput usage
The sub-directory `high-throughput-scripts/` contains scripts for running `toboggan.py` on large data sets, as well as scripts for processing the output of `toboggan.py` on these large data sets into an easily handled format.
See [README.md](./high-throughput-scripts/README.md) for further details.

## Example Usage

The call
```
$>python3 toboggan.py testdata/toboggan_test.graph --indices 1,3-5
```
will run the Toboggan algorithm on the first, third, fourth and fifth graph
contained in `toboggan_test.graph`. The outputs should looks as follows:

```
# No timeout set
# Recovering paths
# Running on instances 1,3-5
# Ground-truth available in file testdata/toboggan_test.truth

File toboggan_test.graph instance 0 name three_paths_10_20_20 with n = 5, m = 6, and truth = 3:
# Preprocessing
# Graph has an edge cut of size 2.
# Investigating cutsets yields bound 3.
# User supplied k value of 3.
# Continuing using k = 3
# Reduced instance has n = 3, m = 4, and lower_bound = 3:
Searching for minimum-sized set of weights, timeout set at -1
#   Call guess_weight with k = 3
# Weights computation took 0.00 seconds
# Solution: {SolvedConstr (10, 20, 20)}
# Now recovering the 3 paths in the solution (10, 20, 20)
# Recovery took 0.00 seconds
# Paths, weights pass test: flow decomposition confirmed.
# Solutions:
#   Path with weight = 10
#   [0, 2, 1, 4]
#   Path with weight = 20
#   [0, 2, 1, 3, 4]
#   Path with weight = 20
#   [0, 1, 3, 4]
Finished instance.


File toboggan_test.graph instance 2 name three_paths_100_200_300 with n = 5, m = 6, and truth = 3:
# Preprocessing
# Graph has an edge cut of size 2.
# Investigating cutsets yields bound 2.
# User supplied k value of 3.
# Continuing using k = 2
# Reduced instance has n = 3, m = 4, and lower_bound = 3:
Searching for minimum-sized set of weights, timeout set at -1
#   Call guess_weight with k = 3
# Weights computation took 0.00 seconds
# Solution: {SolvedConstr (100, 200, 300)}
# Now recovering the 3 paths in the solution (100, 200, 300)
# Recovery took 0.00 seconds
# Paths, weights pass test: flow decomposition confirmed.
# Solutions:
#   Path with weight = 100
#   [0, 1, 3, 4]
#   Path with weight = 200
#   [0, 1, 4]
#   Path with weight = 300
#   [0, 2, 1, 3, 4]
Finished instance.


File toboggan_test.graph instance 3 name four_paths_25_25_50_50 with n = 5, m = 8, and truth = 4:
# Preprocessing
# Graph has an edge cut of size 4.
# Investigating cutsets yields bound 4.
# User supplied k value of 4.
# Continuing using k = 4
# Reduced instance has n = 3, m = 6, and lower_bound = 4:
Searching for minimum-sized set of weights, timeout set at -1
#   Call guess_weight with k = 4
# Weights computation took 0.00 seconds
# Solution: {SolvedConstr (25, 25, 50, 50)}
# Now recovering the 4 paths in the solution (25, 25, 50, 50)
# Recovery took 0.00 seconds
# Paths, weights pass test: flow decomposition confirmed.
# Solutions:
#   Path with weight = 25
#   [0, 1, 4]
#   Path with weight = 25
#   [0, 3, 2, 1, 4]
#   Path with weight = 50
#   [0, 2, 4]
#   Path with weight = 50
#   [0, 3, 4]
Finished instance.


File toboggan_test.graph instance 4 name two_paths_1_8 with n = 4, m = 4, and truth = 2:
# Preprocessing
# Graph has an edge cut of size 2.
# Investigating cutsets yields bound 2.
# User supplied k value of 2.
# Continuing using k = 2
# Reduced instance has n = 2, m = 2, and lower_bound = 2:
Searching for minimum-sized set of weights, timeout set at -1
#   Call guess_weight with k = 2
# Weights computation took 0.00 seconds
# Solution: {SolvedConstr (1, 8)}
# Now recovering the 2 paths in the solution (1, 8)
# Recovery took 0.00 seconds
# Paths, weights pass test: flow decomposition confirmed.
# Solutions:
#   Path with weight = 1
#   [0, 1, 3]
#   Path with weight = 8
#   [0, 2, 3]
Finished instance.
```

The outputs tells us that e.g. the first instance has a solution with one path
of weight 10 and two paths of weight 20. If we run

```
$>python3 toboggan.py testdata/toboggan_test.graph --indices 2
```
the output should consist of the message
```
# No timeout set
# Recovering paths
# Running on instances 2
# Ground-truth available in file testdata/toboggan_test.truth

File toboggan_test.graph instance 1 name trivial_117 with n = 4, m = 3, and truth = 1:
Trivial.
Finished instance.
```

The term `trivial` here means that the instance consists only of a single path.

### Data formatting
#### Datasets

We use two files formats in Toboggan, `.graph` and `.truth`.

A `.graph` file contains an arbitrary number of graphs, stored as follows:

* Each graph-instance begins with a header  line formatted as follows:
  * `# graph number = 0 name = toboggan_test`.
* The line after the header contains the number of vertices in the graph,
* each subsequent line gives the head vertex, tail vertex, and weight for an arc.

A `.truth` can accompany a .graph file and contains the ground-truths (a set of
weighted paths) to the graphs contained in the later. We assume the following convention:

* The file `NAME.truth` contains ground-truth for every graph in `NAME.graph`
* We assume that the ground-truth appear in the same order as their associated graphs
  and they begin with the same header line (see above).
* After the header line, each line corresponds to one of the paths in the solution
* A path is given by a list of integers, separated by spaces. The first integer refers
  to the weight of the path, the subsequent integers are the vertices that appear on the path
  in the order from source to sink.

## Contribution (for Developers)

We welcome contributions to Toboggan, in the form of
[Pull Requests](https://help.github.com/articles/using-pull-requests/),
where you "fork" our repository and then request that we "pull" your changes into the main branch.
You must have a Github account to make a contribution.

Whenever possible, please follow these guidelines for contributions:

- Keep each pull request small and focused on a single feature or bug fix.
- Familiarize yourself with the code base, and follow the formatting
  principles adhered to in the surrounding code (e.g. we are
  [PEP8](https://www.python.org/dev/peps/pep-0008/) compliant).
- We recommend using a linter like
  [flake8](http://flake8.readthedocs.io/en/latest/).

## Citation and License

<!-- **Important**: CONCUSS is *research software*, so you should cite us when you use it in scientific publications! Please see the CITATION file for citation information.
[![DOI](https://zenodo.org/badge/18042/TheoryInPractice/CONCUSS.svg)](https://zenodo.org/badge/latestdoi/18042/TheoryInPractice/CONCUSS) -->

Toboggan is released under the BSD license; see the LICENSE file.
Distribution, modification and redistribution, and incorporation into other
software is allowed.


## Acknowledgements

Development of the Toboggan software package was funded in part by
the [Gordon & Betty Moore Foundation Data-Driven Discovery Initiative](https://www.moore.org/programs/science/data-driven-discovery),
through a [DDD Investigator Award](https://www.moore.org/programs/science/data-driven-discovery/investigators)
to Blair D. Sullivan ([grant GBMF4560](https://www.moore.org/grants/list/GBMF4560)).
