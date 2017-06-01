# FILTER DATA
import collections

def make_tables(inputfile):
    """
    datamatrix -- list of arrays; each array contains info for a single graph instance:
                  datamatrix[j] = [ n, m, n_red, m_red,
                                   k_groundtruth, k_opt, cutset_bound, improved_bound,
                                   t_w, t_path]
                  if toboggan didn't complete, then
                      k_opt = 'None'; t_w = timeout value, t_path = '0.0'
                  if instance is trivial, then
                      n_red = 1, m_red = 0

    datadict -- key = a string containing the filename and instance number of a graph instance,
                     i.e. "[filename] [instance_number]", e.g. "1.graph 1337"
                val = index in datamatrix such that datamatrix[val] = info on that graph instance
    """
    datamatrix = []
    datadict = collections.defaultdict(int)
    idx = 0
    with open(inputfile, 'r') as datafile:
        for line in datafile:
            # a line from datafile has format
            # filename instance# n m n_red m_red k_groundtruth k_opt cutset_bound improved_bound t_w t_path
            temp_line = line.strip().split()
            key = temp_line[0] + ' ' + temp_line[1]
            row = temp_line[2::]
            if datadict[key] == 0:
                datamatrix.append(row)
                datadict[key] = idx
                idx +=1
            else:  # check which row has longer timeout
                newtime = float(row[8])
                oldtime = float(datamatrix[datadict[key]][8])
                if newtime > oldtime:
                    datamatrix[datadict[key]] = row
    return datadict, datamatrix


def print_data_summary( froot, datalen, num_trivial ):
    print("{:s} has \n"
          "\ttotal   instances: {:>12}\n"
          "\ttrivial instances: {:>12}\n"
          "\tnontriv instances: {:>12}\n".format(froot, datalen, num_trivial, datalen-num_trivial))


def print_alg_summary( which_alg, timeoutexceed, num_timedout, num_success ):
    print("{:s} with {}s timeout has\n"
          "\ttimeout instances: {:>12}\n"
          "\tsuccess instances: {:>12}".format( which_alg, timeoutexceed, num_timedout, num_success))