"""
For each dataset, run toboggan (with no weight guessing) on a randomly
sampled subset of instances from that dataset.
"""
import subprocess

froots = ['zebra', 'mouse', 'human', 'salmon']
datadirs = ['zebrafish', 'mouse', 'human', 'salmon']
timeout = 1000

for idx in [0,1,2,3]:
    froot = froots[idx]
    print("Working on {}".format(froot))
    input_dir = '/home/kakloste/research/instances/rnaseq/' + datadirs[idx] + '/'
   
    results_file = 'no-guesses-output-' + froot + '.txt'
    
    # load samples
    samplesfilename = 'random-samples-' + froot + '.txt'
    which_sample = 0
    with open(samplesfilename, 'r') as samplesf:
        for line in samplesf:
            print("{}, {}".format(froot, which_sample))
            which_sample += 1
            
            # get filename and graph index
            parts = line.strip().split()
            graphfilename = str(parts[0]) + '.graph'
            graphindex = int(parts[1])
            
            subprocess.call("python3 toboggan.py {}/{} --skip_truth "
                "--experiment_info --timeout {} --indices {} >>"
                " {}".format(input_dir, graphfilename, timeout, graphindex+1, results_file),
                shell=True)

    print("\t done")
