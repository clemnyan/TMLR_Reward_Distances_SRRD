#!/usr/bin/python3

#read in a file describing a playthrough in ToySteelOcean and generate a csv file for it.

#novice
#Init.Deploy.Check S1,SP,empty.Check S2,SP,empty.Check S3,SP,EB.Move B1,S3.Attack S3,B1,EB.Check S4,B2,ES.Move C,S4.Attack S4,C,EB

#expert

import sys
import os
import copy
import re

#same headings for all test files

headings="T,Reward,Action,Enemy Submarine Health,Enemy Submarine,Enemy Battleship Health,Enemy Battleship,Carrier Health,Carrier,Battleship 1 Health,Battleship 1,Battleship 2 Health,Battleship 2,Scout Plane Health,Scout Plane,Smoke1,Smoke2,Smoke3,Smoke4"
initline="0,0,init,?,concealed,?,concealed,100,idle,100,idle,100,idle,100,undeployed,?,?,?,?"

col={"T":0,"R":1,"A":2,"ESH":3,"ES":4,"EBH":5,"EB":6,"CH":7,"C":8,"B1H":9,"B1":10,"B2H":11,"B2":12,"SPH":13,"SP":14,"S1":15,"S2":16,"S3":17,"S4":18}

def setreward(moves):
    reward=int(0)
    for health in ["CH","B1H","B2H"]:
        if isinstance(moves[health],int) or moves[health]=="100":
            reward+=int(moves[health])
    if isinstance(moves["SPH"],int) or moves[health]=="100":
        reward+=int(int(moves["SPH"])/5)
    for smoke in ["S1","S2","S3","S4"]:
        if moves[smoke] == "empty":
            reward+=int(20)
    for health in ["EBH","ESH"]:
        if isinstance(moves[health],int) or moves[health]=="100":
            reward=reward-int(moves[health])
    moves["R"]=reward


def csvlinetomovedict(csvline):
    csvlist=csvline.split(",")
    if(len(csvlist)!=19):
        return None
    d={}
    doffset=0
    for key in col.keys():
        d[key]=csvlist[doffset]
        doffset=doffset+1
    return d

def movedicttocsvline(movedict):
    outline=""
    for key in col.keys():
        if outline=="":
            outline=movedict[key]
        else:
            outline+=","+str(movedict[key])
    return outline

    

#Define functions that will transform the board state from the previous board state
#need to be wary of multiple changes simultaneously because of multitaskers
def reset(moves):
    moves["R"]=0
    #remove actions that should never be carried over
    moves["A"]=re.sub("init","",moves["A"])
    moves["A"]=re.sub("release scouts","",moves["A"])
#    ships=["B1","B2","SP","C"]
#    for ship in ships:
#        if moves[ship]!="destroyed":
#            moves[ship]="idle"

def deploy(moves,nextmoves,need_reset):
    nextmoves["SPH"]=int(100)
    moves["SP"]="undeployed"
    moves["C"]="deploying scouts"
    moves["A"]="release scouts"
    moves["R"]=int(0)


def attack(moves,nextmoves,line,need_reset):
    if need_reset:
        reset(moves)
        reset(nextmoves)
    vals=line.split(",")
    smoke=vals[0]
    result=vals[len(vals)-1]
    vals=vals[1:-1]
    healthy=True
    for val in vals:
        if(moves[val+"H"]=="0"):
            healthy=False
            print("Bad file, attacking ship has no health")
            sys.exit(-1)
    moves["A"]="attack"
    if healthy:
        for val in vals:
            moves[val]="attack"
    for val in vals:
        nextmoves[val+"H"]=moves[val+"H"]
    if(moves[result+"H"] or moves[result+"H"]==100):
        nextmoves[vals[0]+"H"]=int(max(0,int(moves[vals[0]+"H"])-(100/len(vals))))
    else:
        nextmoves[vals[0]+"H"]=int(max(0,int(moves[vals[0]+"H"])-(90/len(vals))))
    for val in vals:
        if(int(nextmoves[val+"H"])==0):
            nextmoves[val]="destroyed"
        else:
            nextmoves[val]="idle"
    if(isinstance(moves[result+"H"],int)):
        if(moves[result+"H"]-(90*len(vals))>0):
            nextmoves[result+"H"]=moves[result+"H"]-(90*len(vals))
        else:
            nextmoves[result+"H"]=0
            nextmoves[result]="destroyed"
            nextmoves[smoke]="empty"

def moveship(moves,nextmoves,line,need_reset):
    if need_reset:
        reset(moves)
        reset(nextmoves)
    vals=line.split(",")
    smoke=vals[len(vals)-1]
    vals=vals[0:-1]
    moves["A"]="move"
    for val in vals:
        moves[val]="moving"
        nextmoves[val]="idle"

def checksmoke(moves,nextmoves,line,need_reset):
    if need_reset:
        reset(moves)
        reset(nextmoves)
    vals=line.split(",")
    smoke=vals[0]
    result=vals[len(vals)-1]
    vals=vals[1:-1]
    healthy=True
    for val in vals:
        if(moves[val]=="0"):
            healthy=False
            break
    moves["A"]="scout"
    nextmoves["A"]="scout report"
    if healthy:
        for val in vals:
            moves[val]="check smoke"+smoke[1]
    if(result!="empty"):
        if vals[0]=="SP":
            nextmoves["SPH"]=0
            nextmoves["SP"]="destroyed"
            nextmoves["R"]=-5
        else:
            nextmoves[vals[0]+"H"]=int(max(0,int(moves[vals[0]+"H"])-(50/len(vals))))
            if nextmoves[vals[0]+"H"]==0:
                nextmoves[vals[0]+"H"]="destroyed"
            else:
                nextmoves[vals[0]]="idle"
            for val in vals[1:]:
                nextmoves[val+"H"]=moves[val+"H"]
                nextmoves[val]="idle"
        if(result=="EB"):
            nextmoves[smoke]="enemy battleship"
        else:
            nextmoves[smoke]="enemy submarine"
        nextmoves[result]="visible"
        if(moves[result+"H"]=="?"):
            if "SP" in vals and len(vals)==1:
                nextmoves[result+"H"]=100
            else:
                nextmoves[result+"H"]=max(0,100-len(vals)*10)
    else:
        nextmoves[smoke]="empty"
        nextmoves["R"]=20




#movedict will keep track of the moves for the current timestep
#nextmovedict will keep track of the moves for the next timestep
#This is important because columns like reward will be reflected in the next timestep and not the current one
#read in all paragraphs
for arg in sys.argv[1:]:
    count=0
    if not os.path.exists(arg[:-4]):
        os.makedirs(arg[:-4])
    #create actions file
    with open(arg[:-4]+"/actions.csv","w") as actions:
        actions.write("Carrier,Battleship 1,Battleship 2,Scout Plane\n")
    #create attributes file
    with open(arg[:-4]+"/attributes.csv","w") as attributes:
        attributes.write("Enemy Submarine Health,Enemy Submarine,Enemy Battleship Health,Enemy Battleship,Carrier Health,Battleship 1,Battleship 2,Scout Plane Health,Smoke1,Smoke2,Smoke3,Smoke4\n")
    #create sequences and list.files
    with open(arg[:-4]+"/list.files",mode='w') as lfiles:
        with open(arg,mode='r') as infile:
            for line in infile.readlines():
                count=count+1
                print("starting line "+str(count))
                lfiles.write(arg[:-4]+"/"+arg[:-4]+str(count)+".csv\n")
                movedict={}
                nextmovedict={}
                #Write the headings to a test file
                with open(arg[:-4]+"/"+arg[:-4]+str(count)+".csv",mode='w') as outfile:
                    outfile.write(headings+"\n")
                    paragraph=line
                    commands=paragraph.split('.')
                    timenum=1
                    #Iterate over all of the commands
                    for command in commands:
                        movedict=copy.deepcopy(nextmovedict)
                        command=command.strip()
                        print(command)
                        if(command.lower()=="init"):
                            nextmovedict=csvlinetomovedict(initline)
                            movedict=csvlinetomovedict(initline)
                        elif(command.lower()=="deploy"):
                            deploy(movedict,nextmovedict,True)
                        else:
                            moves=command.split(";")
                            for move in moves:
                                if move.startswith("check "):
                                    checksmoke(movedict,nextmovedict,move[6:],True)
                                elif move.startswith("attack "):
                                    attack(movedict,nextmovedict,move[7:],True)
                                elif move.startswith("move "):
                                    moveship(movedict,nextmovedict,move[5:],True)
                                else:
                                    print("unkown move type")
                                    sys.exit(-1)
                        setreward(movedict)
                        outfile.write(movedicttocsvline(movedict)+"\n")
                        timenum=timenum+1
                    setreward(nextmovedict)
                    outfile.write(movedicttocsvline(nextmovedict)+"\n")
                    outfile.write(movedicttocsvline(nextmovedict)+"\n")
