# FILTER DATA
import collections

def get_catfish_tables(directory, inputfile):
    datamatrix = []
    datadict = collections.defaultdict(lambda:-1)
    idx = 0
    with open(directory + inputfile, 'r') as datafile:
        for line in datafile:
            temp_line = line.strip().split()
            key = temp_line[0] + ' ' + temp_line[1]
            row = temp_line[2::]
            if datadict[key] == -1:  # nothing present, so add the row
                datamatrix.append(row)
                datadict[key] = idx
                idx +=1
            else:
                datamatrix[datadict[key]] = row
                    
    return datadict, datamatrix


def make_tables(inputfile):
    """
    datamatrix -- list of arrays; each array contains info for a single graph instance:
                  datamatrix[j] = [ n, m, n_red, m_red,
                                   k_groundtruth, k_opt, cutset_bound, improved_bound,
                                   t_w, t_path, timeout_flag, timeout_limit]
    
                  if toboggan didn't complete, then
                      k_opt = 'None'; t_w = timeout value, t_path = '0.0'
                  if instance is trivial, then
                      n_red = 1, m_red = 0

    datadict -- key = a string containing the filename and instance number of a graph instance,
                     i.e. "[filename] [instance_number]", e.g. "1.graph 1337"
                val = index in datamatrix such that datamatrix[val] = info on that graph instance
    dict_tob_to_cat -- same dictionary as datadict, except the keys are of the format
                           "[filename] [graph_name]", e.g. "1 ESNRUM000345"
                           val = gives the key that this graph instance uses in datadict
    dict_cat_to_tob -- same dictionary as dict_tob_to_cat, except reverse maps vals to keys)
    """
    datamatrix = []
    datadict = collections.defaultdict(lambda:-1)
    idx = 0
    with open(inputfile, 'r') as datafile:
        for line in datafile:
            # a line from datafile has format
            # filename instance# n m n_red m_red k_groundtruth k_opt cutset_bound improved_bound t_w t_path
            temp_line = line.strip().split()
            key = temp_line[0] + ' ' + temp_line[1]
            row = temp_line[2::]
            current_time = float(row[8])
            timeout_limit = float(row[11])
            if (timeout_limit != -1) & (timeout_limit < current_time):
                # if instance failed to timeout, disregard entirely
                continue
            elif datadict[key] == -1:  # nothing present, so add the row
                datamatrix.append(row)
                datadict[key] = idx
                idx +=1
            else:  # row present; check if should be updated
                newtime = float(row[8])
                newstatus = row[10]
                oldtime = float(datamatrix[datadict[key]][8])
                oldstatus = float(datamatrix[datadict[key]][10])
                newlimit = float(row[11])
                oldlimit = float(datamatrix[datadict[key]][11])
                
                if newstatus == '1':
                    if oldstatus == '0':
                        continue
                    elif newlimit < oldlimit:
                        continue
                if (oldstatus == '0') & (newtime > oldtime):
                    continue
                # if we make it this far, replace
                datamatrix[datadict[key]] = row

    # Now create dict of graph file + graph name, from toboggan
    dict_tob_to_cat = collections.defaultdict(lambda:-1)
    toboggandict_counter = collections.defaultdict(int)
    for key, val in datadict.items():
        gname = datamatrix[val][-1]
        gfile = key.split()[0]
        gfile = gfile.split('.')[0]
        newkey = gfile +' ' + gname
        toboggandict_counter[newkey] += 1
        dict_tob_to_cat[newkey] = key

    dict_cat_to_tob = {val: key for key, val in dict_tob_to_cat.items()}
    return datadict, datamatrix, dict_tob_to_cat, dict_cat_to_tob


def print_data_summary( froot, datalen, num_trivial ):
    print("{:s} has \n"
          "\ttotal   instances: {:>12}\n"
          "\ttrivial instances: {:>12}\n"
          "\tnontriv instances: {:>12}\n".format(froot, datalen, num_trivial, datalen-num_trivial))


def print_alg_summary( which_alg, timeoutexceed, num_timedout, num_success ):
    print("{:s} with {}s timeout has\n"
          "\ttimeout instances: {:>12}\n"
          "\tsuccess instances: {:>12}".format( which_alg, timeoutexceed, num_timedout, num_success))


def get_toboggan_timing_info(datadict, datamatrix):
    """
    Outputs a dictionary containing several datastructures of specific info:
            {'num_trivial':num_trivial,
            'num_timedout':num_timedout,
            'time_totals':time_totals,
            'total_num':total_num,
            'nontrivials_dict':nontrivials_dict,
            'toboggan_completed':toboggan_completed,
            'toboggan_timeouts':toboggan_timeouts,
            'toboggan_num_paths_dict':toboggan_num_paths_dict}
        
    toboggan_completed -- dict where key = nontrivial instances on which
                        toboggan successfully terminated, and val = runtime
    toboggan_timeouts -- dict where key = ID of graph instance, val = index in datamatrix
    toboggan_num_paths_dict -- dict with key = ID of graph instance, val = k_opt
    """
    num_trivial = 0
    num_timedout = 0
    time_totals = []
    nontrivials_dict = {}
    toboggan_completed = collections.defaultdict(lambda:-1)
    toboggan_timeouts = collections.defaultdict(lambda:-1)
    total_num = len(datamatrix)
    toboggan_num_paths_dict = collections.defaultdict(lambda:-1)
    for key, val in datadict.items():
        row = datamatrix[val]
        """
        datamatrix[j] = [ n, m, n_red, m_red,
                          k_groundtruth, k_opt, cutset_bound, improved_bound,
                          t_w, t_path, timeout_flag, timeout_limit]
        """
        nontrivials_dict[key] = 1
        if row[2] == '1':  # trivial instance because reduced graph has 1 node
            num_trivial += 1
            nontrivials_dict.pop(key, None)
            toboggan_num_paths_dict[key] = 1
            continue
        elif row[5] == 'None':  # timedout instance
            num_timedout += 1
            toboggan_timeouts[key] = val
            toboggan_num_paths_dict[key] = None
            continue
        elif float(row[11]) < float(row[8]):  # timedout instance that failed to timeout
            # don't count it at all, because we rerun it elsewhere
            print(row[11])
            print(row[8])
            print(key)
            continue
        else:
            time_total = float(row[8]) + float(row[9])
            if time_total == 0.0:
                print("Time of 0.0, something went wrong")
                print(froot + " " + key)
                print(row)
            time_totals.append(time_total)
            toboggan_completed[key] = time_total
            toboggan_num_paths_dict[key] = row[5]
    return {'num_trivial':num_trivial,
            'num_timedout':num_timedout,
            'time_totals':time_totals,
            'total_num':total_num,
            'nontrivials_dict':nontrivials_dict,
            'toboggan_completed':toboggan_completed,
            'toboggan_timeouts':toboggan_timeouts,
            'toboggan_num_paths_dict':toboggan_num_paths_dict}

def get_catfish_timing_info(datadict, datamatrix):
    time_totals = []
    times_dict = {} # collections.defaultdict(lambda:False)
    paths_dict = {} # collections.defaultdict(lambda:False)
    gt_dict = {}
    for key, val in datadict.items():
        row = datamatrix[val]
        """
        datamatrix[j] = [ k_gt, k_catfish, time ]
        """
        if len(row) > 3:
            print(len(row))
        time_totals.append(float(row[2]))
        times_dict[key] = float(row[2])
        paths_dict[key] = int(row[1])
        gt_dict[key] = int(row[0])
    return {
        'time_totals':time_totals,
        'times_dict':times_dict,
        'paths_dict':paths_dict,
        'gt_dict':gt_dict
        }
