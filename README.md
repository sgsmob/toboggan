# Toboggan

Toboggan is a research tool for decomposing a flow on a directed acyclic graph into
a minimal number of paths, a problem that commonly occurs in transcript and metagenomic
assembly.

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
  * `--disprove` - Run instances with parameter k-1 instead of k (needs a `.truth` file)
  * `--indices` INDICES -  Either a file containing indices (position in `.graph`
                     file) on which to run, or a list of indices separated by
                     commas. Ranges are accepted as well, e.g. "1,2-5,6".

## Example Usage

The call
```
$>python3 toboggan.py testdata/toboggan_test.graph --indices 1,3-5
```
will run the Toboggan algorithm on the first, third, fourth and fifth graph
contained in `toboggan_test.graph`. The outputs should looks as follows:

```
# No timeout set
# Running on instances 1,3-5
# Using ground-truth from file testdata/toboggan_test.truth
1 toboggan_test:0 with n = 3, m = 4, and k = 3: 
Computation took 0.00 seconds
Solutions: {SolvedConstr (10, 20, 20)}

3 toboggan_test:2 with n = 3, m = 4, and k = 3: 
Computation took 0.00 seconds
Solutions: {SolvedConstr (100, 200, 300)}

4 toboggan_test:3 with n = 3, m = 6, and k = 4: 
Computation took 0.00 seconds
Solutions: {SolvedConstr (25, 25, 50, 50)}

5 toboggan_test:4 with n = 2, m = 2, and k = 2: 
Computation took 0.00 seconds
Solutions: {SolvedConstr (1, 8)}
```

The outputs tells us that e.g. the first instance has a solution with one path
of weight 10 and two paths of weight 20. If we run 

```
$>python3 toboggan.py testdata/toboggan_test.graph --indices 2
```
the output should consist of the message
```
# No timeout set
# Running on instances 2
# Using ground-truth from file testdata/toboggan_test.truth
2 toboggan_test:1 is trivial.
```

The term `trivial` here mans that the instance consists only of a single path.

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

<!---
**Important**: Toboggan is *research software*, so you should cite us when you use it in scientific publications! Please see the CITATION file for citation information.
[![DOI](https://zenodo.org/...)](https://zenodo.org/badge/...)

--> 

Toboggan is released under the BSD license; see the LICENSE file.
Distribution, modification and redistribution, and incorporation into other
software is allowed.


## Acknowledgements

Development of the Toboggan software package was funded in part by
the [Gordon & Betty Moore Foundation Data-Driven Discovery Initiative](https://www.moore.org/programs/science/data-driven-discovery),
through a [DDD Investigator Award](https://www.moore.org/programs/science/data-driven-discovery/investigators)
to Blair D. Sullivan ([grant GBMF4560](https://www.moore.org/grants/list/GBMF4560)). 




