#!/usr/bin/env python3

import sys
import dtm_irl
import os
import subprocess
from pipes import quote

def buildJoint(dirs):
    joint = list()
    for d in dirs:
        with open(d + "/list.files") as lf:
            lfiles = lf.read().splitlines()
        for f in range(len(lfiles)):
            if not os.path.isabs(lfiles[f]):
                if os.path.exists(d + "/" + lfiles[f]):
                    lfiles[f] = os.path.abspath(d + "/" + lfiles[f])
                else:
                    lfiles[f] = os.path.abspath(d + "/" + os.path.basename(lfiles[f]))
            joint.append(lfiles[f])
    return (joint)

if len(sys.argv) < 9:
    msg = "Usage: " + sys.argv[0] + " <target_directory1,target_directory2,...,target_directoryn>  <others_directory1,...,others_directorym> <target_nbrs_directory1,...target_nbrs_directoryl> <others_nbrs_directory1,...,others_nbrs_directoryk> <neighborhood size >=0> <discount factor [0,1]> <reward cap> <results_directory>\n"
    msg += "\tDTM is built from all directories\n"
    msg += "\tNeighbor trajectories are found in *_nbrs_directory*\n"
    msg += "\tAll trajectory directories assumed to have a list.files\n"
    msg += "\tactions.csv and attributes.csv are taken from target_directory1\n"
    msg += "\tIf neighborhoos size is specified non-zero, then c-neighbors are generated from the target and others directory trajectories.\n"
    sys.exit(msg)

nbrhd_size = int(sys.argv[5])
discount_factor = float(sys.argv[6])
reward_peak = float(sys.argv[7])
results_dir = sys.argv[8]

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

target_list_files = results_dir + "/target_list.files"
with open(target_list_files, mode = 'w') as lf:
    lf.write("\n".join(target_joint)+"\n")

if len(others_joint):
    others_list_files = results_dir + "/others_list.files"
    with open(others_list_files, mode = 'w') as lf:
        lf.write("\n".join(others_joint)+"\n")
else:
    others_list_files = ''

if len(target_nbr_joint):
    target_nbr_list_files = results_dir + "/target_nbr_list.files"
    with open(target_nbr_list_files, mode = 'w') as lf:
        lf.write("\n".join(target_nbr_joint)+"\n")
else:
    target_nbr_list_files = ''

if len(others_nbr_joint):
    others_nbr_list_files = results_dir + "/others_nbr_list.files"
    with open(others_nbr_list_files, mode = 'w') as lf:
        lf.write("\n".join(others_nbr_joint)+"\n")
else:
    others_nbr_list_files = ''

with open(results_dir + "/dtm_list.files", mode = 'w') as lf:
    lf.write("\n".join(target_joint)+"\n")
    if len(others_joint):
        lf.write("\n".join(others_joint)+"\n")
    if len(target_nbr_joint):
        lf.write("\n".join(target_nbr_joint)+"\n")
    if len(others_nbr_joint):
        lf.write("\n".join(others_nbr_joint)+"\n")

subproc = subprocess.Popen(["./DTM_builder","-training",results_dir + "/dtm_list.files","-attributes",attributes_csv,"-actions",actions_csv,"-save",results_dir + "/save.dtm","-diagnostic",results_dir + "/diag"])
rcode = subproc.wait()

if rcode != 0:
    print ("DTM was not constructed, look at {}/diag for log.".format(results_dir))
    sys.exit(rcode)

#run the generated DTM with the original trajectories
dtm_irl.DTM_IRL(dtmfile = results_dir + "/save.dtm", trajectories_file = target_list_files, trajectories_nbrs_file = target_nbr_list_files, evaluate_trajectories_file = others_list_files, evaluate_trajectories_nbrs_file = others_nbr_list_files, actions_csv = actions_csv, max_changes = nbrhd_size, discount_rate = discount_factor, peak = reward_peak, rewards_csv_file = results_dir + "/dtm-output.csv", dtm_report_csvfile = results_dir + "/dtm-report.csv")

