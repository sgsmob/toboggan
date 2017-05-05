"""
Given an output file from run_experiment.py, this script sifts through
the results and reports analysis such as average runtime.

Use:
    python3 data_scrape.py [experimentresults file] [new file to print out to]

    - the experimentresults file should contain output from run_experiments.py
    - the new file will contain a list of instances that timed out
"""

import sys
import matplotlib
# This line prevents an error arising when running this via ssh
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

datafile = sys.argv[1]
timeout_file = sys.argv[2]

files = []
instance_num = []
name = []

timed_out_graphs = []

n_in = []
m_in = []
n_red = []
m_red = []
k_gtrue = []
k_cutset = []
k_improved = []
k_opt = []
time_w = []
time_p = []
time_out = []
weights = []

time_flag = 0


#print "reading in edgelist"
with open(datafile, 'r') as reader:
	content = reader.readlines()
	for line in content:
		if line[0] == "#": # skip that line
			continue
		else:
			parts =  line.strip().split()
			if(len(parts) > 0):
				if(parts[0] == 'File'):
					tmp_file = parts[1]
					tmp_inst = parts[3]
					tmp_name = parts[5]
					files.append(tmp_file)
					instance_num.append(tmp_inst)
					name.append(tmp_name)
				if(parts[0] == 'Timed'):
					timed_out_graphs.append((tmp_file, tmp_inst, tmp_name))
					time_flag = 1
				if(parts[0] == 'All_info'):
					n_in.append(parts[1])
					m_in.append(parts[2])
					n_red.append(parts[3])
					m_red.append(parts[4])
					k_gtrue.append(parts[5])
					k_cutset.append(parts[6])
					k_improved.append(parts[7])
					k_opt.append(parts[8])

					time_w.append(parts[9])
					time_p.append(parts[10])
				if(parts[0] == 'weights'):
					wts = []
					for i in range(1,len(parts)):
						wts.append(int(parts[i]))
					weights.append(wts)
				if(parts[0] == 'Finished'):
					time_out.append(time_flag)
					time_flag = 0


writer = open(timeout_file, 'w')


for t_f, t_i, t_n in timed_out_graphs:
	writer.write("%s %s %s\n" % (t_f, t_i, t_n))

print("Total num instances is {}".format(len(k_opt)))

print("{} timed out graphs written to {}".format(len(timed_out_graphs), timeout_file))


#plot runtime (time_weights + time_paths) as a function of # edges ; as a function of k_optimal
total_times = []
finished_times = []
finished_opt = []
finished_densities = []

for i in range(len(time_w)):
	total_times.append(float(time_w[i]) + float(time_p[i]))
	if time_out[i] == 0:	
		finished_opt.append(int(k_opt[i]))
		finished_times.append(float(time_w[i]) + float(time_p[i]))
		finished_densities.append(float(m_red[i])/float(n_red[i]))


plt.plot(finished_opt, finished_times,  'ro')
plt.xlim([-1,10])
plt.ylim([-0.5,5])
plt.savefig('time_from_opt.png')
plt.clf()

plt.plot(finished_densities, finished_times,  'ro')
plt.xlim([-1,10])
plt.ylim([-0.5,5])
plt.savefig('time_from_density.png')

# look at average runtime for non-trivial, non-timeouts
nontt_times = 0.0
div = 0
for i in range(len(k_opt)):
	if time_out[i]==0 and int(k_opt[i])>1:
		nontt_times = nontt_times + total_times[i]
		div = div + 1

print("(3) Number of non-trivial, non-timeouts is {}, ave runtime is {}".format(div, nontt_times/div))

#look at distribution of gaps between lower_bound and k_optimal
gaps = []
cutsetgaps = []
for i in range(len(k_opt)):
	if(time_out[i] == 0):
                gaps.append(int(k_opt[i]) - int(k_improved[i]))
                cutsetgaps.append(int(k_opt[i]) - int(k_cutset[i]))

print("(4) Distribution of gaps between lower_bound and k_optimal (gap-size: frequency):")
print(Counter(gaps))
print("Distribution of gaps between cutset and k_optimal (gap-size: frequency):")
print(Counter(cutsetgaps))

#scan for any instances where k_optimal != k_groundtruth
print("(1) Instances where k-optimal != k-groundtruth:")
counter = 0
for i in range(len(k_opt)):
	if k_opt[i] != k_gtrue[i] and time_out[i] == 0:
            print("{} -- {} -- {} : {}, {}".format(files[i], instance_num[i], name[i], k_opt[i], k_gtrue[i]))
            counter += 1
print("{} instances where k-optimal < ground-truth".format(counter))















