#!/usr/bin/env python3

import sys
import dtm_irl

if len(sys.argv) < 2:
    print("Usage: "+sys.argv[0]+" <directory> [<neighborhood size (default 4)> <discount factor (default 1.0)> <reward cap (default 100)>]")
    print("\tArguments contained in brackets are optional")
    print("\tOutput will go in <directory>/dtmirl_output.csv")
    sys.exit(-1)

runnames=sys.argv[1]
trajfile=runnames+"/list.files"
numtraj=len(runnames.split(','))

print(numtraj)

neighborhood_size=4
if len(sys.argv) >2:
    neighborhood_size=int(sys.argv[2])

discount_factor=1.0
if len(sys.argv) >3:
    discount_factor=int(sys.argv[3])

magnitude_cap=100
if len(sys.argv) >4:
    magnitude_cap=int(sys.argv[4])

if numtraj > 1:
    for rundir in runnames.split(','):
        #dtm_irl.DTM_IRL(dtmfile=".tmp_dtm_joint",trajectories_file=rundir+"/list.files",actions_csv=rundir+"/actions.csv",rewards_csv_file=rundir+"/dtmirl_output.csv",max_changes=neighborhood_size,discount_rate=discount_factor,peak=magnitude_cap,dtm_report_csvfile=rundir+"/report.csv",evaluate_trajectories_file=".tmp_traj_joint")
        dtm_irl.DTM_IRL(dtmfile=".tmp_dtm_joint",trajectories_file=rundir+"/list.files",actions_csv=rundir+"/actions.csv",rewards_csv_file=rundir+"/pairwise-output.csv",max_changes=neighborhood_size,discount_rate=discount_factor,peak=magnitude_cap,dtm_report_csvfile=rundir+"/pairwise-report.csv",evaluate_trajectories_file=[s+"/list.files" for s in runnames.split(",")])
        print("----------Commander "+rundir+" trajectory analysis----------")
        dtm_irl.DTM_trajectoryAnalysis(".tmp_dtm_joint",rundir+"/list.files",rundir+"/actions.csv")
        print("----------End Commander "+rundir+" trajectory analysis----------")
else:
    rundir=runnames
    dtm_irl.DTM_IRL(dtmfile=rundir+"/save",trajectories_file=rundir+"/list.files",actions_csv=rundir+"/actions.csv",rewards_csv_file=rundir+"/dtmirl_output.csv",max_changes=neighborhood_size,discount_rate=discount_factor,peak=magnitude_cap,dtm_report_csvfile=rundir+"/report.csv")
    print("----------Commander "+rundir+" trajectory analysis----------")
    dtm_irl.DTM_trajectoryAnalysis(rundir+"/save",rundir+"/list.files",rundir+"/actions.csv")
    print("----------End Commander "+rundir+" trajectory analysis----------")
#dtmfile, trajectories_file, actions_csv, rewards_csv_file, max_changes = 5, discount_rate = 1.0, peak = 100.0, traj_prod_sum = False, traj_sum = False, incremental_prod_sum_aggregate = True, path_emphasis = False, dtm_report_csvfile='', evaluate_trajectories_file=''):

