# FILTER DATA
import collections

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
    """
    datamatrix = []
    #datadict = collections.defaultdict(int)
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


def get_toboggan_timing_info(datadict, datamatrix):
    num_trivial = 0
    num_timedout = 0
    time_totals = []
    total_num = len(datamatrix)
    for key, val in datadict.items():
        row = datamatrix[val]
        """
        datamatrix[j] = [ n, m, n_red, m_red,
                          k_groundtruth, k_opt, cutset_bound, improved_bound,
                          t_w, t_path, timeout_flag, timeout_limit]
        """
        if row[2] == '1':  # trivial instance because reduced graph has 1 node
            num_trivial += 1
            continue
        elif row[5] == 'None':  # timedout instance
            num_timedout += 1
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
    return {'num_trivial':num_trivial,
            'num_timedout':num_timedout,
            'time_totals':time_totals,
            'total_num':total_num}