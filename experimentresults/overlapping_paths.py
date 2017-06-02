import os
import shutil
import argparse
import subprocess


#this code will take in 1 file containing a list of instances, for each run catfish and toboggan
#look at the path outputs

def call_tob(input_file,  results_file):
    with open(results_file, 'w') as f:
        out = subprocess.run("python ../toboggan.py {} "
                             "--skip_truth "
                             "".format(input_file, results_file),
                              stdout=subprocess.PIPE, shell=True)
        # print(out.stdout.decode())
        f.write(out.stdout.decode())

def call_cf(path_to_catfish, input_file, output_file):
    subprocess.run("{} -i {} -o {}".format(path_to_catfish, input_file,
                                            output_file),
                   stdout=subprocess.DEVNULL, shell=True) 

def iterate_over_input_list(input_list,input_dir,path_to_catfish):

    tob_results = os.path.join(os.getcwd(),".tob_results.txt")
    cf_results = os.path.join(os.getcwd(),".cf_results.txt")
                            
    with open(input_list, 'r') as instancelist:
        for line in instancelist:
            #print(line)
            if line[0] == "#":  # skip that line
                continue
            parts = line.strip().split()
            if (len(parts) == 0):
                continue
            name = "{}.{}.sgr".format(*parts)
            filename = os.path.join(input_dir, name)
            print(os.path.basename(filename))

            call_tob(filename, tob_results)
            call_cf(path_to_catfish, filename, cf_results)


            #read contents of temp files
            tob_paths = tob_output_parser(tob_results)
            cf_paths = cf_output_parser(cf_results)

            #contrast temp files
            print("intersection size: {} toboggan size: {} catfish size: {}"
                  "\n".format(len(tob_paths & cf_paths),
                              len(tob_paths),
                              len(cf_paths)) )

def tob_output_parser(tob_results):

    paths_found = False
    tob_paths = set()
    
    with open(tob_results, 'r') as reader:
        for line in reader:
            if line.startswith("#") and "Solutions:" in line:
                paths_found = True
            elif paths_found == True and '[' in line:
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
















