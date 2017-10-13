# Toboggan experiments


## Results files (in ./data)

* `edge-flows-[name].txt` --- one file per dataset; each row contains a graph instance key [filename, graph index], and all flow values for a graph instance.
* `path-weights-[name].txt` --- one file per dataset; each row contains a graph instance key [filename, graph index], and all path-weights for toboggan's solution for that graph instance.


* `[name]-master-file.txt` --- all, verbatim toboggan output for that dataset
* `all-[name].txt` --- bare-bones info on output of toboggan: 1 row per instance. Produce this using `/high-throughput-scripts/file_scrape.py`
* `master-clean-[name].txt` --- all pathsets output by toboggan (no weights). Produce using `/high-throughput-scripts/decomposition_scrape.py`


## Script purposes

* `figure_groundtruth_recovery` --
*
