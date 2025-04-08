#!/usr/bin/env python3

import sys
import pydtm
import pandas as pd
import graphviz
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import csv
from sklearn import tree
import lime
import lime.lime_tabular
import pathlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout':True})
import pdfkit

dtm=None
state_report=None
types=None

if len(sys.argv)<3:
    sys.exit()

dtm=pydtm.pydtm(sys.argv[2]+"/unmerged/"+sys.argv[1]+"/joint-learned.dtm")
state_report=pd.read_csv(sys.argv[2]+"/unmerged/"+sys.argv[1]+"/state_report.csv")

#drop targets and state index
state_report=state_report.iloc[:,1:int(len(state_report.columns)/2)]

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

rewards=df["reward"]
df=df.drop("reward",axis=1)


#list columns that need to be one hot encoded
onecols=[x for x in range(len(df.columns)) if df.iloc[:,x].dtype == np.object]

ohenc=OneHotEncoder()
lbenc=LabelBinarizer()

#Perform one-hot encoding for strings
for i in onecols:
    data=df.iloc[:,int(i)]
    lbenc.fit(list([str(x) for x in list(data)]))
    newhead=["{0}_{1}".format(header[i],x) for x in lbenc.classes_]
    #labelbinarizer stupidly decides when it's binary to return an entirely
    #different format, so when we get an error, assume that's the problem
    binarized=lbenc.transform(list(data))
    try:
        onehot=pd.DataFrame(binarized,columns=newhead)
    except:
        binarized=np.hstack((binarized,1-binarized))
        onehot=pd.DataFrame(binarized,columns=newhead)

    df=df.join(onehot)

#remove columns that we have one-hot encoded
df=df.drop(df.columns[onecols],axis=1)


#Begin classifying rewards
clf = tree.DecisionTreeClassifier()
clf=clf.fit(df,rewards)
dot_data=tree.export_graphviz(clf,out_file=None,feature_names=df.columns)
graph=graphviz.Source(dot_data)
graph.render(sys.argv[2]+"/"+sys.argv[1]+"-out")

dot_data=tree.export_graphviz(clf,out_file=None,feature_names=df.columns,max_depth=7)
graph=graphviz.Source(dot_data)
graph.render(sys.argv[2]+"/"+sys.argv[1]+"-short")

inds=clf.feature_importances_>0
print(*sorted(list(zip(df.columns[inds],clf.feature_importances_[inds])),key=lambda x: x[1],reverse=True),sep='\n')

#load the trajectories by importing Doc's dtm_irl functions
exec(open('dtm_irl.py', "rb").read())
trajectories=DTM_loadTrajectoriesFile(dtm,sys.argv[1]+'/list.files',sys.argv[1]+'/actions.csv')
trajmap=[[(trajectories[j][i-1],trajectories[j][i],trajectories[j][i+1]) for i in range(1,int(len(trajectories[j])),2)] for j in range(len(trajectories))]
traj_trip_inds=[[trip for trip in range(len(dtm.triples)) for i in traj if i == (dtm.triples[trip].source,dtm.triples[trip].action,dtm.triples[trip].dest)] for traj in trajmap]

#Build the explainer from LIME
lime_train=df.as_matrix()
dtree_pred=lambda x:clf.predict_proba(x)

explainer=lime.lime_tabular.LimeTabularExplainer(lime_train,feature_names=df.columns,class_names=["reward={0}".format(i) for i in clf.classes_])

exp=explainer.explain_instance(lime_train[1,:],predict_fn=dtree_pred)
pathlib.Path(sys.argv[2]+"/explanation/dtree").mkdir(parents=True,exist_ok=True)

for traj in range(len(traj_trip_inds)):
    for act_trip_ind in range(len(traj_trip_inds[traj])):
        exp=explainer.explain_instance(lime_train[traj_trip_inds[traj][act_trip_ind],:],predict_fn=dtree_pred,top_labels=3)
        ht=exp.as_html()
        pdfkit.from_string(ht,sys.argv[2]+"/explanation/dtree/traj["+str(traj)+"]["+str(act_trip_ind)+"].pdf")
        #fig=exp.as_pyplot_figure()
        #fig.savefig(sys.argv[2]+"/explanation/dtree/traj["+str(traj)+"]["+str(act_trip_ind)+"].png")
        #matplotlib.pyplot.close(fig)

