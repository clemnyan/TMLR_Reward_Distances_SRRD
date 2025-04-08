#!/usr/bin/env python3.7

import sys
import L_MDP
import RoMDP
import FoMDP
import os
import gzip
BARON = "/home/esj/bin/baron-lin64/baron"
BARONLICE = "/home/esj/bin/baron-lin64/baronlice.txt"
CPLEX = "/home/esj/bin/CPLEX_Studio1210/cplex/bin/x86-64_linux/cplex"

num_classes = 2
baron_max_time = 10000
simplify_masking = True #False
feasibility_only = False

if len(sys.argv) < 5:
    msg = "Usage: " + sys.argv[0] + " <action.csv attributes_file, target_list_file, nontarget_list_file, output dir>"
    sys.exit(msg)

action_file=sys.argv[1]
attributes_file=sys.argv[2]
target_file=sys.argv[3]
nontarget_file=sys.argv[4]
out_dir=sys.argv[5]

print( "Action=",action_file)
print( "Attributes=",attributes_file)
print( "Target=",target_file)
print( "Non_target=",nontarget_file)
print( "Output=",out_dir)
L_MDP.learnMDP(action_file, attributes_file, target_file, None, None, nontarget_file, out_dir.format(num_classes, simplify_masking), 1, 10, FoMDP.constructOptimization, "FoMDP", "GUROBI", None, None, None, num_classes, simplify_masking, feasibility_only)
#L_MDP.learnMDP(action_file, attributes_file, target_file, None, None, nontarget_file, out_dir.format(num_classes, simplify_masking), 1, 10, FoMDP.constructOptimization, "FoMDP", "BARON", BARON, baron_max_time, BARONLICE, num_classes, simplify_masking, feasibility_only)
