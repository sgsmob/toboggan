import os
import shutil
import argparse
import subprocess


#this code will take in 1 file containing a list of instances, for each run catfish and toboggan
#look at the path outputs

def call_tob(input_file,  results_file):
	subprocess.call("python3 ../toboggan.py {}  "
                    "--skip_truth --experiment_info  > {}.txt"
                    "".format(input_file, results_file), shell=True)

def call_cf(path_to_catfish, input_file, output_file):
	subprocess.call("{} -i {} -o {}".format(path_to_catfish, input_file, output_file), shell=True) 

def iterate_over_input_list(input_list,input_dir,path_to_catfish):

	tob_results = ".tob_results.txt"
	cf_results = ".cf_results.txt"
                            
	with open(input_list, 'r') as instancelist:
		for line in instancelist:
		    print(line)
		    if line[0] == "#":  # skip that line
			continue
		    parts = line.strip().split()
		    if(len(parts) > 0):
			filename = os.path.join(input_dir, parts[0])
			print(filename)


		call_tob(filename, tob_results)
		call_cf(path_to_catfish, filename, cf_results)


		#read contents of temp files
		tob_paths = tob_output_parser(".tob_results.txt")
		cf_paths = cf_output_parser(".cf_results.txt")

		#contrast temp files
		print("intersection size: {} toboggan size: {} catfish size: {}".format(len(tob_paths & cf_paths), len(tob_paths), len(cf_paths)) )

def tob_output_parser(tob_results):

	paths_found = False
	tob_paths = set()
	
	with open(tob_results, 'r') as reader:
		for line in reader:
			if line == "# Solutions:":
				paths_found = True
			if paths_found == True:
				if line.startswith("# 	Path with weight ="):
					continue
				tob_paths.add(line[1:].strip())
	return tob_paths
			
			

def cf_output_parser(cf_results):

	cf_paths = set()

	with open(cf_results, 'r') as reader:
		for line in reader:
			edited_line = line.strip().split()
			path_string = "[" + ", ".join(edited_line[7:]) + "]"
			cf_paths.add(path_string)

	return cf_paths

#run the instance on toboggan
#catch output in temp file
#run the instance on catfish
#catch output in temp file

parser = argparse.ArgumentParser()
parser.add_argument("input_list", type=str, help="list of files to iterate over")
parser.add_argument("input_dir", type=str, help="directory containing graph and truth files")
parser.add_argument("path_to_catfish", type=str, help="path to catfish")
args = parser.parse_args()



iterate_over_input_list(args.input_list, args.input_dir, args.path_to_catfish)
















