# Experiments for Toboggan

Explaining scripts used in running experiments, as well as their various output files.

* `run_experiments.py` -- use this to run toboggan.py on every instance contained in every file in a specified directory, with the ability to specify a timeout value.
	* pipe the output to a text file named `output-[dataname]-[timeout value].txt`, e.g. `output-human-5.txt`.
* `data_scrape.py` -- reads in the output of `run_experiments.py` and does a few things:
	* reports average runtime and a few other statistics
	* outputs a list of instances where ground truth was larger than optimal size 
	* outputs a list of instances that timed out (formatted so that later scripts can re run just those instances)
* `run_timedout_experiments.py` -- reads in the output from `data_scrape.py` (list of timed out instnaces) and re-runs toboggan.py with new timedout parameter setting.
	* pipe the output to a text file name `output-timeouts-[dataname]-[timeout value].txt`
