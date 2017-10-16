# Explanation of No Guesses experiments

This folder contains a version of toboggan altered so that it does not guess any weights. The rest of the files / algorithm / pipeline is the same as the standard toboggan code.


To reproduce this experiment, execute the following:

1. `randomly-sample-instances.ipnyb` --- this script samples without replacement until it has created a list of instances (length is user-specified) that have at least 2 paths, and such that toboggan already successfully terminated on them. Output saved in `random-samples-[name].txt`

2. `random-sampling.py` --- call this on any file `random-samples-[name].txt` to run toboggan on each of the instances listed in the file. All the output from toboggan will be written to the file `no-guesses-output-[name].txt`

3. Run `file_scrape.py` (from the official toboggan repo) on the output files `no-guesses-output-[name].txt` to generate the condensed stats for each instance. Output file named `condensed-stats-[name].txt`
