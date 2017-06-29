# Executing Toboggan in a high-throughput setting

This sub-directory contains codes intended to make it convenient to run `toboggan.py` on many graphs and process the output easily.

## Codes for executing experiments

Codes:
  * `run_on_directory.py` -- will execute `toboggan.py` on all files in a specified directory, and record output in a single file.
  * `run_on_list.py` -- will execute `toboggan.py` on all instances whose filename and graph ID are listed in an input txt file, and record all output in a single file.

Command line usage:

```
$>python3 run_on_directory.py [path to data sets] [output.txt] [timeout limit]
```
This will run `toboggan.py` on every instance in every file in the specified directory, and record the output of each such execution in `output.txt`.
This script will terminate any individual call to `toboggan.py` if the weight computation phase exceeds `timeout limit` seconds.

```
$>python3 run_on_directory.py [path to data sets] [input.txt] [output.txt] [timeout limit]
```
The script `run_on_list.py` executes `toboggan.py` on every instance listed in `input.txt`, and record the output of each such execution in `output.txt`. This script handles `timeout limit` the same as above.
The file `example_list_of_timeouts.txt` shows how to properly format a txt file for use with `run_on_directory.py`: namely, on each line have `filename instancenumber` for a single instance.


## Codes for processing output of experiments
The above codes produce text files containing the results from running `toboggan.py` on many instances. The codes described here make it easier to process the information contained in those `output.txt` files so it can be analyzed.

* `file_scrape.py` -- parses the files `output.txt` produced by the codes above, and writes formatted information to `processed_output.tsv` for easier handling. Command line Usage:
```
$>python3 file_scrape.py [output.txt] [processed_output.tsv]
```

The output of `file_scrape.py` contains one line for each graph instance; each line is a tab-separated list of information summarizing the performance of toboggan on a single graph instance.
  The information in each line is:

  ```
    [filename, instancenumber,
     n_input, m_input, n_reduced, m_reduced,
     k_groundtruth, cutset_bound, improved_bound, k_optimal,
     time_weights, time_path, timeout_flag, timeout_limit,
     graphname]
  ```
    * `filename`, `instance number`, and `graphname` are identifier information for the graph instance taken from the input file.
    * `n` denotes the number of nodes in the instance, `m` the number of edges.
    * `k_groundtruth` is the number of paths in the groundtruth solution and `k_optimal` is the number of paths in the solution identified by toboggan; it equals None if toboggan times out on the instance. `cutset_bound` and `improved_bound` are lowerbounds for `k_optimal` that are computed in pre-processing.
    * `time_weights` and `time_paths` give the amount of time required for the weight-computation and path-recovery phases of toboggan, respectively.
    * `timeout_flag` equals 1 if toboggan timed out and 0 otherwise. `timeout_limit` indicates the timeout limit with which toboggan was called; it gets switched to -1 if the graph is trivial.


* `decomposition_scrape.py` -- parses the files `output.txt` produced by the codes above, and writes just the pathset information to `formatted_pathsets.txt` for easier handling. Command line Usage:
```
$>python3 decomposition_scrape.py [output.txt] [formatted_pathsets.txt]
```

## Example usage

To test `run_on_directory.py`, do the following:

1. From the project's root directory, execute python on the repo's test datasets by calling `python3 ./high-throughput-scripts/run_on_directory.py ./testdata toboggan_output.txt 1`. This will produce an output file `toboggan_output.txt` in the project's root directory.

2. From `./high-throughput-scripts/` call `python3 file_scrape.py ../toboggan_output.txt processed_output.txt`. This will produce a file `processed_output.txt` in `./high-throughput-scripts/` that lists the performance of toboggan on each testdata set.

3. From `./high-throughput-scripts/` call `python3 decomposition_scrape.py ../toboggan_output.txt processed_paths.txt`. This will produce a file `processed_paths.txt` in `./high-throughput-scripts/` containing just the paths from toboggan's output.


To test `run_on_list.py`, do the following:

1. From the project's root directory, call `python3 ./high-throughput-scripts/run_on_list.py ./testdata ./high-throughput-scripts/example_list_of_instances.txt output_toboggan.txt 20`. This will produce an output file `output_toboggan.txt` in the project's root directory that should closely resemble the following:
```
# Timeout is set to 20
# Recovering paths
# Running on instance(s) 3
# Ground-truth available in file ./testdata/toboggan_test.truth

File toboggan_test.graph instance 2 name three_paths_100_200_300 with n = 5, m = 6, and truth = 3:
# Preprocessing
#	Graph has an edge cut of size 2.
#	Investigating cutsets yields bound 2.
#	User supplied k value of 1.
#	Continuing using k = 2
# Reduced instance has n = 3, m = 4, and lower_bound = 2:
Searching for minimum-sized set of weights, timeout set at 20
# 	Call guess_weight with k = 2
# 	Call guess_weight with k = 3
# Weights computation took 0.00 seconds
# Solution: {SolvedConstr (100, 200, 300)}
#	Now recovering the 3 paths in the solution (100, 200, 300)
# Recovery took 0.00 seconds
# Paths, weights pass test: flow decomposition confirmed.
# Solutions:
# 	Path with weight = 100
# 	[0, 1, 3, 4]
# 	Path with weight = 200
# 	[0, 1, 4]
# 	Path with weight = 300
# 	[0, 2, 1, 3, 4]
# All_info	n_in	m_in	n_red	m_red	k_gtrue	k_cut	k_impro	k_opt	time_w	time_p
All_info	5	6	3	4	3	2	2	3	0.0014390945434570
312	0.00013947486877441406
weights	 100 200 300
Finished instance.

# Timeout is set to 20
# Recovering paths
# Running on instance(s) 4
# Ground-truth available in file ./testdata/toboggan_test.truth

File toboggan_test.graph instance 3 name four_paths_25_25_50_50 with n = 5, m = 8, and truth = 4:
# Preprocessing
#	Graph has an edge cut of size 4.
#	Investigating cutsets yields bound 4.
#	User supplied k value of 1.
#	Continuing using k = 4
# Reduced instance has n = 3, m = 6, and lower_bound = 4:
Searching for minimum-sized set of weights, timeout set at 20
# 	Call guess_weight with k = 4
# Weights computation took 0.00 seconds
# Solution: {SolvedConstr (25, 25, 50, 50)}
#	Now recovering the 4 paths in the solution (25, 25, 50, 50)
# Recovery took 0.00 seconds
# Paths, weights pass test: flow decomposition confirmed.
# Solutions:
# 	Path with weight = 25
# 	[0, 1, 4]
# 	Path with weight = 25
# 	[0, 3, 2, 1, 4]
# 	Path with weight = 50
# 	[0, 2, 4]
# 	Path with weight = 50
# 	[0, 3, 4]
# All_info	n_in	m_in	n_red	m_red	k_gtrue	k_cut	k_impro	k_opt	time_w	time_p
All_info	5	8	3	6	4	4	4	4	0.0010094642639160
156	0.0003139972686767578
weights	 25 25 50 50
Finished instance.
```

2. Next, from `./high-throughput-scripts/` call `python3 file_scrape.py ../output_toboggan.txt processed_output.txt`. This will produce a file `processed_output.txt` in `./high-throughput-scripts/` that should match the following, except for the floating point numbers in columns 11 and 12:
```
toboggan_test.graph	2	5	6	3	4	3	2	2	3	0.0014390945	0.0001394749	0	20	three_paths_100_200_300
toboggan_test.graph	3	5	8	3	6	4	4	4	4	0.0010094643	0.0003139973	0	20	four_paths_25_25_50_50
```

3. Finally, from `./high-throughput-scripts/` call `python3 decomposition_scrape.py ../output_toboggan.txt processed_paths.txt`. This will produce a file `processed_paths.txt` in `./high-throughput-scripts/` that should exactly match the following
```
# toboggan_test 2
0 2 1 3 4
0 1 3 4
0 1 4
# toboggan_test 3
0 2 4
0 3 4
0 3 2 1 4
0 1 4
```
