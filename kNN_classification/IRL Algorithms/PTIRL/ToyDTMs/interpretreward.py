#!/usr/bin/env python3

import sys
import pydtm
import pandas as pd
import graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import csv
from sklearn import tree

dtm=None
state_report=None
types=None

if len(sys.argv)>2:
    dtm=pydtm.pydtm(sys.argv[2]+"/unmerged/Scenarios/joint-learned.dtm")
    state_report=pd.read_csv(sys.argv[2]+"/unmerged/Scenarios/state_report.csv")
    #read in columns file (containing datatypes
    with open(sys.argv[1]+"/floatintother.csv",'r') as csvfile:
        types=list(csv.reader(csvfile))[1]
else:
    dtm=pydtm.pydtm("scenout/unmerged/Scenarios/joint-learned.dtm")
    state_report=pd.read_csv("scenout/unmerged/Scenarios/state_report.csv")
    #read in columns file (containing datatypes
    with open("Scenarios/floatintother.csv",'r') as csvfile:
        types=list(csv.reader(csvfile))[1]



#drop targets and state index
state_report=state_report.iloc[:,1:len(state_report.columns)/2]

#rename columns to start_ and dest_
header=["start_{0}".format(i) for i in list(state_report.columns)]
header.append("action")
header.extend(["dest_{0}".format(i) for i in list(state_report.columns)])
header.append("reward")

#read all of the (state,action,state) triples and their rewards and put them in
#a list of lists
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


rewards=df["reward"]
df=df.drop("reward",axis=1)


#list columns that need to be one hot encoded
onecols=[i for i,x in enumerate(types) if x == 'other'] #source
onecols.append(len(types)) #action
onecols.extend([len(types)+1+i for i in onecols[0:len(onecols)-1]]) #dest

ohenc=OneHotEncoder()
lbenc=LabelBinarizer()

#Perform one-hot encoding for strings
print(onecols)
#for i in reversed(onecols):
for i in onecols:
    print(i)
    print(df.iloc[:,i])
    lbenc.fit(list([str(x) for x in df.iloc[:,i]]))
    print(lbenc.classes_)
    newhead=["{0}_{1}".format(header[i],x) for x in lbenc.classes_]
    onehot=pd.DataFrame(lbenc.transform(list(df.iloc[:,i])),columns=newhead)
    df=df.join(onehot)

#remove columns that we have one-hot encoded
df=df.drop(df.columns[onecols],axis=1)


#Begin classifying rewards
clf = tree.DecisionTreeClassifier()
clf=clf.fit(df,rewards)
dot_data=tree.export_graphviz(clf,out_file=None,feature_names=df.columns)
graph=graphviz.Source(dot_data)
if len(sys.argv)>2:
    graph.render(sys.argv[1]+"-out")
else:
    graph.render("Scenario")

