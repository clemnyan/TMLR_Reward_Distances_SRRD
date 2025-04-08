#!/usr/bin/env python3.8

import sys
import L_MDP_analytics
import time
import compress_pickle

if len(sys.argv) < 4:
    msg = "Usage: " + sys.argv[0] + " <num_dirs n> <descriptor 1> ... <descriptor n> <results directory 1> ... <results directory n> <analytics output fn>\n"
    msg += "\t-- Recommended to place descriptors and directories in \"s\n" 
    msg += "\t-- Best to have compressed pickled analytics files\n"
    sys.exit(msg)

num_dirs = int(sys.argv[1])
if len(sys.argv) < 3 + 2 * num_dirs:
    msg = "Usage: " + sys.argv[0] + " <num_dirs n> <descriptor 1> ... <descriptor n> <results directory 1> ... <results directory n> <analytics output fn>\n"
    msg += "\t-- Recommended to place descriptors and directories in \"s\n" 
    msg += "\t-- Best to have compressed pickled analytics files\n"
    sys.exit(msg)

descriptors = [ sys.argv[idx] for idx in range(2,2+num_dirs) ]
print ('Descriptors = ', end='')
print (descriptors)

dirs = [ sys.argv[idx] for idx in range(2+num_dirs, 2+2*num_dirs) ]
print ('Corresponding Directories = ', end='')
print (dirs)
analytics_fn = sys.argv[2+2*num_dirs]
print ('Analytics output file = ', end='')
print (analytics_fn)
start_time = time.time()
print ('Performing analytics...')
r = L_MDP_analytics.analyze(descriptors, dirs)
print ('Saving compressed pickled analytic results...')
with open (analytics_fn, "wb") as analytics:
    compress_pickle.dump(r, analytics)
    analytics.close()

print ('********************Elapsed time = {}'.format(time.time() - start_time))
