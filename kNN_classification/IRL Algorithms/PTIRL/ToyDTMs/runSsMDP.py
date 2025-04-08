#!/usr/bin/env python3.7

import sys
import L_MDP
import SsMDP
import os
import gzip
BARON = "/home/esj/bin/baron-lin64/baron"
BARONLICE = "/home/esj/bin/baron-lin64/baronlice.txt"
CPLEX = "/home/esj/bin/CPLEX_Studio1210/cplex/bin/x86-64_linux/cplex"

def buildJoint(dirs):
    joint = list()
    for d in dirs:
        if os.path.exists(d + "/list.files.gz"):
            lf = gzip.open(d + "/list.files.gz", 'rt')
        elif os.path.exists(d + "/list.files"):
            lf = open(d + "/list.files", 'rt')
        lfiles = lf.read().splitlines()
        for f in range(len(lfiles)):
            if not os.path.isabs(lfiles[f]):
                if os.path.exists(d + "/" + lfiles[f]):
                    lfiles[f] = os.path.abspath(d + "/" + lfiles[f])
                else:
                    lfiles[f] = os.path.abspath(d + "/" + os.path.basename(lfiles[f]))
            joint.append(lfiles[f])
        lf.close()
    return (joint)

print(sys.argv)

if len(sys.argv) < 10:

    msg = "Usage: " + sys.argv[0] + " <target_directory1,target_directory2,...,target_directoryn>  <others_directory1,...,others_directorym> <target_nbrs_directory1,...target_nbrs_directoryl> <others_nbrs_directory1,...,others_nbrs_directoryk> <reward cap> <tau> <optimizer> <results_directory_prefix> <feasibility_only>\n"
    msg += "\tDTM/MDP is built from all directories\n"
    msg += "\tNeighbor trajectories are found in *_nbrs_directory*\n"
    msg += "\tAll trajectory directories assumed to have a list.files\n"
    msg += "\tactions.csv and attributes.csv are taken from target_directory1\n"
    msg += "\t<optimizer> can be either \'BARON\' or \'CPLEX\'\n"
    msg += "\t<tau> is the maximum number of hidden states\n"
    msg += '\t<feasibility_only> can be set to "True" or "False"\n'
    sys.exit(msg)


reward_peak = float(sys.argv[5])
tau = int(sys.argv[6])
optimizer = sys.argv[7]
base_dir = sys.argv[8]
results_dir = sys.argv[8] + '-' + optimizer
if sys.argv[9] == 'True':
    feasibility_only = True
else:
    feasibility_only = False

print ('Target directories = ', end='')
target_dirs = sys.argv[1].split(',')
print (target_dirs)
target_joint = buildJoint(target_dirs)

print ('Others directories = ', end='')
if sys.argv[2] == '':
    others_dirs = list()
else:
    others_dirs = sys.argv[2].split(',')
print (others_dirs)
others_joint = buildJoint(others_dirs)

print ('Target neighbor directories = ', end='')
if sys.argv[3] == '':
    target_nbr_dirs = list()
else:
    target_nbr_dirs = sys.argv[3].split(',')
print (target_nbr_dirs)
target_nbr_joint = buildJoint(target_nbr_dirs)

print ('Others neighbor directories = ', end='')
if sys.argv[4] == '':
    others_nbr_dirs = list()
else:
    others_nbr_dirs = sys.argv[4].split(',')
print (others_nbr_dirs)
others_nbr_joint = buildJoint(others_nbr_dirs)

actions_csv = target_dirs[0] + '/actions.csv'
attributes_csv = target_dirs[0] + '/attributes.csv'

os.makedirs(results_dir, exist_ok = True)

target_list_files = results_dir + "/target_list.files.gz"
with gzip.open(target_list_files, mode = 'wt') as lf:
    lf.write("\n".join(target_joint)+"\n")

if len(others_joint):
    others_list_files = results_dir + "/others_list.files.gz"
    with gzip.open(others_list_files, mode = 'wt') as lf:
        lf.write("\n".join(others_joint)+"\n")
else:
    others_list_files = None

if len(target_nbr_joint):
    target_nbr_list_files = results_dir + "/target_nbr_list.files.gz"
    with gzip.open(target_nbr_list_files, mode = 'wt') as lf:
        lf.write("\n".join(target_nbr_joint)+"\n")
else:
    target_nbr_list_files = None

if len(others_nbr_joint):
    others_nbr_list_files = results_dir + "/others_nbr_list.files.gz"
    with gzip.open(others_nbr_list_files, mode = 'wt') as lf:
        lf.write("\n".join(others_nbr_joint)+"\n")
else:
    others_nbr_list_files = None

with gzip.open(results_dir + "/dtm_list.files.gz", mode = 'wt') as lf:
    lf.write("\n".join(target_joint)+"\n")
    if len(others_joint):
        lf.write("\n".join(others_joint)+"\n")
    if len(target_nbr_joint):
        lf.write("\n".join(target_nbr_joint)+"\n")
    if len(others_nbr_joint):
        lf.write("\n".join(others_nbr_joint)+"\n")

if optimizer == "BARON":
    exe_fn = BARON
    license_fn = BARONLICE
elif optimizer == "CPLEX":
    exe_fn = CPLEX
    license_fn = None
else:
    exe_fn = None
    license_fn = None

#run the generated DTM with the original trajectories
L_MDP.learnMDP(action_spec_csvfilename = actions_csv, state_spec_csvfilename = attributes_csv, target_traj_list_fn = target_list_files, target_nbr_traj_list_fn = target_nbr_list_files, other_nbr_traj_list_fn = others_nbr_list_files, other_traj_list_fn = others_list_files, descriptor = base_dir, peak = reward_peak, optimizer = optimizer, exe_fn = exe_fn, maxtime = 10000, license_fn = license_fn, tau = 2, constructOptimization = SsMDP.constructOptimization, constructDescriptor = "SsMDP", num_classes = None, simplify_masking = False, feasibility_only = feasibility_only)
