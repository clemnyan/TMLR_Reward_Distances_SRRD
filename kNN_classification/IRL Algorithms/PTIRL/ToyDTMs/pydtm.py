#!/usr/bin/env python3

import sys
import math
import csv
import os
import subprocess

class triple:
#    source=None
#    action=None
#    dest=None
#    weight=None
#    prob=None
#    reward=None

    def __init__(self,lines = None):
        if lines == None:
            self.source = None
            self.action = None
            self.dest = None
            self.weight = None
            self.prob = None
            self.reward = None
        else:
            vals=lines[0].split()

            if vals[0]=="None":
                self.source=None
            else:
                self.source=int(vals[0])

            if vals[1]=="None":
                self.action=None
            else:
                self.action=int(vals[1])

            if vals[2]=="None":
                self.dest=None
            else:
                self.dest=int(vals[2])

            if vals[3]=="None":
                self.weight=None
            else:
                self.weight=float(vals[3])

            if vals[4]=="None":
                self.prob=None
            else:
                self.prob=float(vals[4])

            if vals[5]=="None":
                self.reward=None
            else:
                self.reward=float(vals[5])


class dtma:
#    human_name=None
#    action_name=None
    def __init__(self,lines=None):
        if lines is not None:
            self.human_name=lines[0]
            self.action_name=lines[1]
            if self.action_name[-1] == '\n':
                self.action_name = self.action_name[:-1]
            self.num_params=int(lines[2])
            if self.num_params !=0:
                print("don't know how to handle action parameters")
                sys.exit(-1)

class dtmcs:
#    les_set=[]
#    csname=None
    def buildKey(self):
        if self.modified:
            self.key = ''
            for lidx, value in enumerate(self.les_set):
                if int(value) == 1:
                    self.key += str(lidx) + '+'
            self.modified = False

    def __init__(self,lines=None):
        if lines is not None:
            self.csname=lines[0]
            self.les_set=lines[1].split()
            self.modified = True
            self.buildKey()

class dtmle:
#    attributes={}
    def buildKey(self):
        if self.modified:
            self.key = ''
            for key, value in sorted(self.attributes.items()):
                self.key += '\"' + value + '\"'
            for key, value in sorted(self.target_attributes.items()):
                self.key += '\"' + value + '\"'
            self.modified = False

    def __init__(self,lines=None):
        if lines is not None:
            self.attributes = {}
            self.target_attributes = {}
            if int(lines[0]) > 0:
                li=lines[1].split()[1:]
                self.attributes={li[i]:li[i+1] for i in range(0,len(li),2)}
                li=lines[2].split()[1:]
                self.target_attributes={li[i]:li[i+1] for i in range(0,len(li),2)}
            self.modified = True
            self.buildKey()

class dtmlespace:
#    num_le=0
#    lines_read=0
#    max_number_attributes=0
#    max_number_attribute_values=0
#    offset=0
#    les=[]
#    num_attributes=0
#    attribute_names=[]
#    attribute_vals=[]
    def buildDicts(self):
        if self.modified:
            self.attribute_names_dict = {}
            self.attribute_values_dict = [0] * self.num_attributes
            for aidx, attrib in enumerate(self.attribute_names):
                self.attribute_names_dict[attrib] = aidx
                self.attribute_values_dict[aidx] = {}
                for vidx, value in enumerate(self.attribute_vals[aidx]):
                    self.attribute_values_dict[aidx][value] = vidx

            self.le_keys = {}
            for lidx, le in enumerate(self.les):
                self.le_keys[le.key] = lidx
            self.modified = False

    def save(self,outfile):
        outfile.write(str(self.max_number_attributes)+"\n")
        outfile.write(str(self.max_number_attribute_values)+"\n")
        outfile.write(str(self.num_le)+"\n")
        outfile.write(str(self.num_attributes)+"\n")
        for att in self.attribute_names:
            outfile.write(str(len(att))+"\n")
            outfile.write(att+"\n")
        for att in self.attribute_names:
            #for val in self.attribute_vals[att]:
            outfile.write(str(len(self.attribute_vals[self.attribute_names_dict[att]]))+"\n")
            for val in self.attribute_vals[self.attribute_names_dict[att]]:
                outfile.write(str(len(val))+"\n")
                outfile.write(val+"\n")
        for le_ind in range(0,self.num_le):
            le=self.les[le_ind]
            outfile.write(str(len(le.attributes))+"\n")
            outfile.write(str(len(le.attributes))+' '+' '.join([str(key)+" "+str(value) for key, value in sorted(le.attributes.items(),key=lambda i: int(i[0]))])+" \n")
            outfile.write(str(len(le.target_attributes))+' '+"\n")
            if(len(le.target_attributes)>0):
                outfile.write(str(len(le.target_attributes))+' '+' '.join([str(key)+" "+str(value) for key, value in sorted(le.target_attributes.items(),key=lambda i: int(i[0]))])+" \n")

    def __init__(self,lines=None):
        self.les = list()
        if lines is not None:
            self.max_number_attributes=int(lines[0])
            self.max_number_attribute_values=int(lines[1])
            self.num_le=int(lines[2])
            self.num_attributes=int(lines[3])
            self.offset=4
            self.attribute_names = list()
            self.attribute_vals = list()
            for att in range(0,self.num_attributes):
                strlen=int(lines[self.offset])
                self.attribute_names.append(lines[self.offset+1][:strlen])
                self.offset+=2
            for att in range(0,self.num_attributes):
                num_local_atts=int(lines[self.offset])
                self.offset+=1
                self.attribute_vals.append([])
                for att_val_ind in range(0,num_local_atts):
                    strlen=int(lines[self.offset])
                    self.attribute_vals[att].append(lines[self.offset+1][:strlen])
                    self.offset+=2
            for le_ind in range(0,self.num_le):
                le=dtmle(lines[self.offset:])
                if int(lines[self.offset]) > 0:
                    self.offset +=3
                else:
                    self.offset +=1
                self.les.append(le)
            self.modified = True
            self.buildDicts()

    def findAttributeIndex(self, attribute):
        try:
            aidx = self.attribute_names_dict[attribute]
        except KeyError:
            return (-1)
        return (aidx)

    def findAttributeValueIndex(self, attribute_index, value):
        try:
            vidx = self.attribute_values_dict[attribute_index][value]
        except KeyError:
            return (-1)
        return (vidx)

    def getLE(self, key):
        try:
            lidx = self.le_keys[key]
        except KeyError:
            return (-1)
        return(lidx)


class pydtm:
#    key_length=0
#    num_cs=0
#    num_a=0
#    a_maxkeylength=0
#    cs_maxkeylength=0
#    lespace=None
#    states=[]
#    actions=[]
#    triples=[]

    # Builds dictionaries for fast access
    def buildDicts(self):
        if self.modified:
            self.states_dictionary = {}
            self.actions_dictionary = {}
            for aidx, action in enumerate(self.actions):
                self.actions_dictionary[action.action_name] = aidx
            for csidx, cs in enumerate(self.states):
                self.states_dictionary[cs.key] = csidx
        self.modified = False

    def __init__(self,inpath=None):
        self.states = list()
        self.actions = list()
        self.triples = list()
        lines=[]
        if inpath is None:
            self.num_cs=0
            self.num_a=0
            self.a_maxkeylength=512
            self.cs_maxkeylength=512
            self.key_length=512
        else:
            self.from_saved_cformat(inpath)

    def build_from_folder(self,rundir):
        subproc=subprocess.Popen([os.getenv("HOME")+"/src/C++/DTM/DTMBuilder/DTM_builder","-training",rundir+"/list.files","-attributes",rundir+"/attributes.csv","-actions",rundir+"/actions.csv","-diagnostic",rundir+"/diag","-save",rundir+"/save.dtm"])
        subproc.wait()
        self.__init__(rundir+"/save.dtm")

    def from_python(self,cogstates,trajs):
        #assumes one learning episode per state
        self.num_cs=len(cogstates)
        self.lespace=lespace()

        #needed for lespace
        attribute_names=set()
        attribute_values={}

        #this line just generates an identity matrix to use for les
        tmp_les=[["0"]*i+["1"]+["0"]*max(0,(self.num_cs-1)-i) for i in range(self.num_cs)]

        #build list of attributes
        max_atts=0
        for i in range(self.num_cs):
            attribute_names=attribute_names.union(cogstates[i].keys())
            max_atts=max(max_atts,len(cogstates[i].keys()))

        attribute_names=list(attribute_names)
        self.lespace.max_number_attributes=max_atts
        self.lespace.num_attributes=len(attribute_names)
        self.lespace.attribute_names = attribute_names

        self.lespace.attribute_names_dict = {}
        for aidx, attrib in enumerate(dtm.lespace.attribute_names):
            self.lespace.attribute_names_dict[attrib] = aidx

        #build list of attribute values per attribute for lespace
        self.lespace.attribute_vals = [set()] * dtm.lespace.num_attributes
        for i in range(self.num_cs):
            for att in cogstates[i].keys():
                if att in attribute_values:
                    self.lespace.attribute_vals[attribute_names.index(att)]=self.lespace.attribute_vals[attribute_names.index(att)].union(list(cogstates[i][att]))
        for i in range(len(attribute_names)):
            self.lespace.attribute_vals[i]=list(self.lespace.attribute_vals[i])
        

        self.lespace.attribute_values_dict = [0] * dtm.lespace.num_attributes
        for aidx in range(self.lespace.num_attributes):
            self.lespace.attribute_values_dict[aidx] = {}
            for vidx, value in enumerate(self.lespace.attribute_vals[aidx]):
                self.lespace.attribute_values_dict[aidx][value] = vidx
        self.lespace.max_number_attribute_values=max([len(self.lespace.attribute_vals[i]) for i in range(self.lespace.num_attributes)])

        for i in range(self.num_cs):
            #create all of the cognitive states
            #assumes that the state names are empty
            state=dtmcs()
            state.csname='\n'
            state.les_set=tmp_les[i]
            state.modified = True
            state.buildKey()
            self.states.append(state)

            #create all of the learning episodes
            le=dtmle()
            le.attributes = cogstates[i]
            le.target_attributes = {}
            le.modified = True
            le.buildKey()
            self.lespace.les.append(le)

        self.lespace.num_le=len(self.lespace.les)
        self.lespace.le_keys = {}
        for lidx, le in enumerate(self.lespace.les):
            self.lespace.le_keys[le.key] = lidx
        self.lespace.modified = False

        #construct trajectories assuming state indices match up with cogstates
        #and action indices are also consistent
        trips={}
        doubs={}
        for traj in trajs:
            for i in range(0,len(traj)-2,2):
                act_ind=0

                if traj[i+1] in [action.action_name for action in self.actions]:
                    act_ind=[action.action_name for action in self.actions].index(traj[i+1])
                else:
                    act_ind=len(self.actions)
                    action=dtma()
                    action.human_name='\n'
                    action.action_name=traj[i+1]
                    action.num_params=0
                    self.actions.append(action)

                trip=(traj[i],act_ind,traj[i+2]) 
                if trip in trips:
                    trips[trip]=trips[trip]+1.0
                else:
                    trips[trip]=0.0

                doub=(traj[i],act_ind)
                if doub in doubs:
                    doubs[doub]=doubs[doub]+1.0
                else:
                    doubs[doub]=0.0
                
        for trip,weight in trips.items():
            n_trip=triple()
            n_trip.source = trip[0]
            n_trip.action = trip[1]
            n_trip.dest = trip[2]
            n_trip.weight = weight
            n_trip.prob = weight/doubles[(trip[0],trip[1])]
            self.triples.append(n_trip)

    def from_saved_cformat(self,inpath):
        with open(inpath) as infile:
            lines=infile.readlines()
        self.a_maxkeylength=int(lines[0])
        self.cs_maxkeylength=int(lines[1])
        self.num_cs=int(lines[2])
        self.num_a=int(lines[3])
        self.key_length=max(self.a_maxkeylength,self.cs_maxkeylength)
        offset=4
        self.lespace=dtmlespace(lines[offset:])
        offset+=self.lespace.offset
        for cs in range(0,self.num_cs):
            self.states.append(dtmcs(lines[offset:]))
            offset+=2
        for a in range(0,self.num_a):
            self.actions.append(dtma(lines[offset:]))
            offset+=3
        num_triples=int(lines[offset])
        offset+=1
        for trip in range(0,num_triples):
            self.triples.append(triple(lines[offset:]))
            offset+=1
        self.modified = True # indicates if there has been a change to the dtm that warrants an update of the dictionaries
        self.buildDicts()

    def findActionIndex(self, action):
        try:
            aidx = self.actions_dictionary[action]
        except KeyError:
            return (-1)
        return(aidx)

    def getCS(self, key):
        try:
            csidx = self.states_dictionary[key]
        except KeyError:
            return (-1)
        return (csidx)

    def __str__(self):
        return "Key length = "+str(self.key_length)+"\nNum Cognitive States = "+str(self.num_cs)+"\nNum Actions = "+str(self.num_a)+"\nMax Action Key Length = "+str(self.a_maxkeylength)+"\nMax Cognitive State Key Length = "+str(self.cs_maxkeylength)+"\nNum Learning Episodes = "+str(self.lespace.num_le)+"\nNum Cognitive States = "+str(len(self.states))+"\nNum Actions = "+str(len(self.actions))+"\nNum Triples = "+str(len(self.triples))

    def toDOT(self,outpath):
        with open(outpath,mode='w') as outfile:
            outfile.write("digraph g{ node[shape=record];\n")
            for trip in self.triples:
                outfile.write(str(trip.source)+"->"+str(trip.dest)+"[label=\"a="+str(trip.action)+",p="+str(trip.prob)+",r="+str(trip.reward)+"\"];\n")
                #outfile.write(str(trip.source)+"->"+str(trip.dest)+"[label=\""+str(trip.action)+"|"+str(trip.weight)+"|"+str(trip.prob)+"|"+str(trip.reward)+"\"];\n")
            outfile.write("}\n")


    def save(self,outpath):
        with open(outpath,mode='w') as outfile:
            outfile.write(str(self.a_maxkeylength)+"\n")
            outfile.write(str(self.cs_maxkeylength)+"\n")
            outfile.write(str(self.num_cs)+"\n")
            outfile.write(str(self.num_a)+"\n")
            self.lespace.save(outfile)
            for state in self.states:
                outfile.write(state.csname)
                outfile.write(' '.join(state.les_set)+" \n")
            for action in self.actions:
                outfile.write(action.human_name)
                outfile.write(action.action_name+"\n")
                outfile.write(str(action.num_params)+"\n")
            outfile.write(str(len(self.triples))+"\n")
            for trip in self.triples:
                outfile.write(str(trip.source)+" "+str(trip.action)+" "+ str(trip.dest)+" "+ str(trip.weight)+" "+str(trip.prob)+ " "+str(trip.reward)+"\n")

    def generateStateReport(self, report_csv_filename):
        with open(report_csv_filename, 'w') as report:
            writer = csv.writer(report)
            rows = list()
            header = list()
            header.append('State Idx')
            for a_idx in range(self.lespace.num_attributes): # attributes
                header.append(self.lespace.attribute_names[a_idx])
            header.append('Target')
            for a_idx in range(self.lespace.num_attributes):
                header.append(self.lespace.attribute_names[a_idx]) # traget attributes
            rows.append(header)
            for cs_idx, state in enumerate(self.states):
#                print ('cs_idx = ', end='')
#                print (cs_idx)
                row = list()
                row.append(cs_idx)
                cs_key = state.key
                le_idx = int(cs_key[:-1])
#                print ('le_idx = ', end='')
#                print (le_idx)
                le = self.lespace.les[le_idx]
                for a_idx in range(self.lespace.num_attributes):
                    try:
                        val = le.attributes[str(a_idx)]
#                        print (val)
                        row.append(self.lespace.attribute_vals[a_idx][int(val)])
                    except KeyError:
                        row.append(' ')
                row.append('Target')
                for a_idx in range(self.lespace.num_attributes):
                    try:
                        val = le.target_attributes[a_idx]
                        row.append(self.lespace.attribute_vals[a_idx][val])
                    except KeyError:
                        row.append(' ')
                rows.append(row)
            writer.writerows(rows)

def getComplexity(dtm):
    #compute complexity of the DTM graph structure
    #compute complexity of individual nodes
    #compute complexity of the learning episodes
    #compute complexity of the DTM reward distribution
    pass
