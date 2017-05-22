# Toboggan experiment notes

## Unexplained timeout behavior

Running on zebrafish/83.graph instance 9941 with a timeout of 30 set, toboggan.py nevertheless ran for 41 seconds (at which point the instance successfully terminated). To the best of my memory, this took place on shrubbery while multiple other experiments (i.e. calling toboggan.py) were taking place in parallel via screen.

Output:
```
Searching for minimum-sized set of weights, timeout set at 30
#     Call guess_weight with k = 3
#     Call guess_weight with k = 4
#     Call guess_weight with k = 5
#     Call guess_weight with k = 6
# Weights computation took 41.10 seconds
```

One proposed cause was that the signals that the timeout function uses could interfere with the signals of the other experiments' calls to toboggan.py -- this seemed unlikely, though, because this signal interference should happen only when the multiple signals are run on the same processor via multi-threading, which we have not been doing.

Another proposed cause was that some low-level function (like a numpy or scipy call to C code for solving linear systems?) was running long and could not be interrupted by python's signal module.

To investigate, I re-ran toboggany.py on the same instance, this time with no other experiments running in parallel.

* results running with no timeout are in `investigating-timeout-wo-timeout.txt`
* results running with timeout=30 are in `investigating-timeout-to30.txt`

These results showed toboggan.py successfully timedout on this instance in 30s when the timeout was used; when the timeout was not used, toboggan.py ran for 41s and successfully terminated.

Looking over the profiler output for the two function calls, nothing seems obviously out of place.


##  Runtime comparison on zebrafish

Mike and Kyle changed the way toboggan.py checked for linear dependence of new constraints, and wanted to test the difference in runtime. So we compiled a list of zebra instances on which old and new toboggan both terminated in under 15 seconds (about 300 instances in the list). We ran old and new toboggan on each instance (all in serial) and found roughly a factor 5.5x speed-up.

This was partly motivated by the fact that our first rounds of experiments with the new toboggan ("rref toboggan", because we implemented reduced row echelon form) showed rref-toboggan was 2x faster on small instances, but *slower* on larger instances. We suspected this was because we ran so many experiments in parallel on shrubbery via screen. This runtime-comparison experiment seems to confirm that suspicion, so now experiments are run in serial.

* runtime-comparison-zebra.txt -- contains list of zebra instances on which we performed the comparison
* analysis-compare-zebra-new.txt -- analysis of rref-toboggan
* analysis-compare-zebra-old.txt -- analysis of old toboggan
