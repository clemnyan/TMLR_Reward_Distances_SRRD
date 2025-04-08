#!/usr/bin/env python3

import sys
import pydtm
import pandas as pd

dtm=pydtm.pydtm(sys.argv[1])
state_report=pd.read_csv(sys.argv[2])
#drop targets and state index
state_report=state_report.iloc[:,1:len(state_report.columns)/2]

header=["start_{0}".format(i) for i in list(state_report.columns)]
header.append("action")
header.extend(["dest_{0}".format(i) for i in list(state_report.columns)])
header.append("reward")

mat=[]
rews=[]
for trip in dtm.triples:
    row=list(state_report.iloc[int(trip.source),:])
    row.append(trip.action)
    row.extend(state_report.iloc[int(trip.dest),:])
    row.append(trip.reward)
    mat.append(row)

df=pd.DataFrame(mat,columns=header)
print(df)

