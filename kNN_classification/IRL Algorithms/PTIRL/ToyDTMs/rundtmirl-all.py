#!/usr/bin/env python3

import sys
import dtm_irl
import dtm_neighborhood
import dtm_states
import pydtm

if len(sys.argv) < 2:
    print("Usage: "+sys.argv[0]+" <directory> <branching (True/False)> <merged states (True/False)> <merged % (default is 90)> [<discount factor (default 1.0)> <reward cap (default 100)>]")
    print("\tArguments contained in brackets are optional")
    print("\tOutput will go in <directory>/dtmirl_output.csv")
    sys.exit(-1)

runnames=sys.argv[1]

branching=False
if len(sys.argv) >2:
    branching=bool(sys.argv[2])

merged_s= False
if len(sys.argv) >3:
    merged_s=bool(sys.argv[3])

merged_pct = 90.0
if len(sys.argv) >4:
    merged_pct = float(sys.argv[4])

neighborhood_size=4
if len(sys.argv) >5:
    neighborhood_size=int(sys.argv[5])

discount_factor=1.0
if len(sys.argv) >6:
    discount_factor=int(sys.argv[6])

magnitude_cap=100
if len(sys.argv) >7:
    magnitude_cap=int(sys.argv[7])


dtm = pydtm.pydtm('.tmp_dtm_joint')
action_dir = runnames.split(',')[0] # just take first one
graph = dtm_neighborhood.DTM_makeGraph(dtm)

if merged_s == True: # Do the merged states on dtm
    print ('Merging states in DTM based on DTM trajectories used in building...')
    trajs = dtm_irl.DTM_loadTrajectoriesFile(dtm, '.tmp_traj_joint', action_dir+"/actions.csv")
    (dtm, graph) = dtm_states.DTM_computeMergeStates(dtm, graph, trajs, percent_distant_pairs = merged_pct)

    # NOTE -- Merging only happens in the base DTM. Merging is NOT called again for the training trajectories so as to avoid double merging

dist_ = dtm_states.DTM_allPairsShortestPath(graph, dtm.num_cs, dtm.num_a)

#if numtraj > 1:
for rundir in runnames.split(','):
    dtm_irl.DTM_IRL(dtmfile=dtm,trajectories_file=rundir+"/list.files",actions_csv=rundir+"/actions.csv",rewards_csv_file=rundir+"/pairwise-merged-output.csv",discount_rate=discount_factor,peak=magnitude_cap,dtm_report_csvfile=rundir+"/pairwise-merged-report.csv",evaluate_trajectories_file=[s+"/list.files" for s in runnames.split(",")],merged_states=False,nbr_branching=branching,max_changes=neighborhood_size)
    print("----------Commander "+rundir+" trajectory analysis----------")
    dtm.generateStateReport(rundir+"/state_report.csv")
    dtm_irl.DTM_trajectoryAnalysis(dtm,rundir+"/list.files",rundir+"/actions.csv", dist=dist_)
    print("\tJoint commanders trajectory analysis ---")
    dtm_irl.DTM_trajectoryAnalysis(dtm,".tmp_traj_joint",rundir+"/actions.csv", dist=dist_)
    print("----------End Commander "+rundir+" trajectory analysis----------")
    dtm.save(rundir+"/joint-learned.dtm")
    dtm.toDOT(rundir+"/pairwise.dot")
#else:
#    rundir=runnames
#    dtm_irl.DTM_IRL(dtmfile=dtm,trajectories_file=rundir+"/list.files",actions_csv=rundir+"/actions.csv",rewards_csv_file=rundir+"/dtmirl_merged_output.csv",max_changes=neighborhood_size,discount_rate=discount_factor,peak=magnitude_cap,dtm_report_csvfile=rundir+"/dtmirl_merged_report.csv",merged_states=False,nbr_branching=branching)
#    dtm_irl.DTM_IRL(dtmfile=rundir+"/save",trajectories_file=rundir+"/list.files",actions_csv=rundir+"/actions.csv",rewards_csv_file=rundir+"/dtmirl_output.csv",max_changes=neighborhood_size,discount_rate=discount_factor,peak=magnitude_cap,dtm_report_csvfile=rundir+"/report.csv")
#dtmfile, trajectories_file, actions_csv, rewards_csv_file, max_changes = 5, discount_rate = 1.0, peak = 100.0, traj_prod_sum = False, traj_sum = False, incremental_prod_sum_aggregate = True, path_emphasis = False, dtm_report_csvfile='', evaluate_trajectories_file=''):

