#!/usr/bin/python3

import sys
import os
import math

if(len(sys.argv)<5):
    print("Usage: "+sys.argv[0]+" outputdirectory |actions| |state1| |state2| ...")
    print("ex: "+sys.argv[0]+" testdir 2 2 2")
    sys.exit(-1)

csvpath=sys.argv[1]
if not os.path.exists(csvpath):
    os.makedirs(csvpath)

numactions=int(sys.argv[2])

maxcomb=1
for i in sys.argv[3:]:
    maxcomb=maxcomb*int(i)

heading='humanname,action'
for i in range(3,len(sys.argv)):
    heading=heading+","+str(int(i-3))

def tocsvline(act,stat):
    line=','+str(int(act))
    val=stat
    for i in range(3,len(sys.argv)):
        numvals=int(sys.argv[i])
        line=line+","+str(int(val%numvals))
        val=int(val/numvals)

    return(line)
    


seqlength=maxcomb-1
#create attributes file
with open(csvpath+"/attributes.csv","w") as attribs:
    attribs.write("0")
    for i in range(4,len(sys.argv)):
        attribs.write(","+str(int(i-3)))
    attribs.write("\n")

#create list of files to run
lfiles=open(csvpath+"/list.files","w")

#iterate over the total number of sequences possible
for cur in range(0,math.factorial(seqlength*numactions)):
    lfiles.write(csvpath+"/"+str(int(cur))+".csv\n")
    seqnum=cur
    #open the file for writing and give it the heading
    with open(csvpath+'/'+str(int(seqnum))+'.csv','w') as f:
        f.write(heading+"\n")
        #Generate the sequence
        state=int(seqnum%maxcomb)
        seqnum=int(seqnum/maxcomb)

        action=int(seqnum%numactions)
        seqnum=int(seqnum/numactions)

        f.write("0"+tocsvline(action,state)+"\n");
        for s in range(1,seqlength):
            state=int(seqnum%maxcomb)
            seqnum=int(seqnum/maxcomb)

            action=int(seqnum%numactions)
            seqnum=int(seqnum/numactions)

            f.write(str(s)+tocsvline(action,state)+"\n");

lfiles.close()

#create actions file
with open(csvpath+"/actions.csv","w") as actions:
    actions.write("action\n")
