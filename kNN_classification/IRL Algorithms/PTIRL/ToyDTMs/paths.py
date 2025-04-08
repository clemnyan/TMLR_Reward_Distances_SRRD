#!/usr/bin/env python3

import networkx as nx
import re
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats


arguments=sys.argv[1:]
edgepattern='"([A-Z]+)"->"([A-Z]+)"\[label="(.*)\|.*\|([^"]+)".*'
#edgepattern='"([A-Z]+)"->"([A-Z]+)".*\|.*\|([^"]+)".*'
nodepattern='"([A-Z]+)"\[.*\|([^"]+).*'
nodepatternnoreward='"([A-Z]+)"\[.*'

edgere=re.compile(edgepattern)
nodere=re.compile(nodepattern)
nodenorewardre=re.compile(nodepatternnoreward)

dg=nx.DiGraph()

for arg in arguments:
    basepath="./"
    if arg.rfind("/")>0:
        basepath=arg[:arg.rfind("/")+1]

    has_act=False
    if not os.path.exists(arg):
        continue

    filename=arg
    actname=""
    savedir=arg+"/res"

    if os.path.isdir(arg):
        filename=arg+"/save1.dot"
        basepath=arg
    else:
        savedir=arg[:-3]+"res"

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    linenum=0
#read .dot file and make a graph called dg
#with open("test.dg",mode='r') as gvfile:
    with open(filename,mode='r') as gvfile:
        for line in gvfile.readlines():
            linenum=linenum+1
            edgematch=edgere.search(line)
            nodematch=nodere.search(line)
            nodenorewardmatch=nodenorewardre.search(line)
            if(edgematch):
                node1=edgematch.group(1)
                node2=edgematch.group(2)
                action=edgematch.group(3)
                prob=float(edgematch.group(4))
                #print("matched an edge "+node1+" "+node2+" "+str(prob))
                if action not in dg:
                    dg.add_node(node1+"|"+action)
                    dg.node[node1+"|"+action]['act_reward']=0.0
                    dg.node[node1+"|"+action]['reward']=0.0
                if(node1 in dg and node2 in dg):
                    #print("added edge")
                    dg.add_edge(node1,node1+"|"+action,weight=prob)
                    dg.add_edge(node1+"|"+action,node2,weight=1.0)
                else:
                    if(node1 in dg):
                        print(node2+" not found on line "+str(linenum))
                    else:
                        print(node1+" not found on line "+str(linenum))
            elif(nodematch):
                node=nodematch.group(1)
                reward=float(nodematch.group(2))
                #print(line+": "+str(reward))
                if(not node in dg):
                    dg.add_node(node,reward=reward)
                    #print("adding node: "+node+" with reward: "+str(reward))
                else:
                    dg.node[node]['reward']=reward
                    #print("updating reward for node to: "+str(reward))
            elif(nodenorewardmatch):
                node=nodenorewardmatch.group(1)
                if(not node in dg):
                    dg.add_node(node)
                    #print("adding node "+node+" without a reward")
            else:
                #print("encountered some sort of error: "+line)
                pass
#add "actual" rewards from initial dtm
    if os.path.exists(arg+"/save.dot"):
        has_act=True
        with open(arg+"/save.dot",mode='r') as gvfile:
            for line in gvfile.readlines():
                linenum=linenum+1
                nodematch=nodere.search(line)
                if(nodematch):
                    node=nodematch.group(1)
                    reward=float(nodematch.group(2))
                    if(node in dg):
                        dg.node[node]['act_reward']=reward

    #initialize all nodes that don't have a reward
    for node in dg.nodes():
        if "reward" not in dg.node[node].keys():
            dg.node[node]['reward']=0.0
        if "act_reward" not in dg.node[node].keys():
            dg.node[node]['act_reward']=0.0

    distances={}
    with open(savedir+"/paths.csv",'w') as out:
        #print headings
        out.write("Source/Dest,"+",".join(dg.nodes())+"\n")
        #compute paths between nodes in dg
        for node1 in dg.nodes():
            distances[node1]={}
            line=node1
            for node2 in dg.nodes():
                distance=0
                paths=nx.all_simple_paths(dg,node1,node2)
                for path in paths:
                    prob=1.0
                    reward=0
                    for i in range(0,len(path)-1):
                        prob=prob*dg[path[i]][path[i+1]]['weight']
                        reward=reward+dg.node[path[i+1]]['reward']
                    distance=distance+(prob*reward)
                line=",".join([line,str(distance)])
                distances[node1][node2]=distance
            out.write(line+"\n")

#make variables so that when this is run inside of python and not from the
#script that the variables stay in the environment
    betweenness=nx.betweenness_centrality(dg)
    close_vitality=nx.closeness_vitality(dg,weight='weight')
    for b in betweenness:
        if "|" in b:
            continue
        dg.node[b]['betweenness']=betweenness[b]
        dg.node[b]['closeness_vitality']=close_vitality[b]
    del close_vitality
    del betweenness

    rewards=[dg.node[node]['reward'] for node in dg.nodes() if "|" not in node]

#with open(sys.argv[1]+"/__temp_reward.csv",mode='r') as rewardfile:
#    next(rewardfile)
#    rewards=[float(reward) for reward in rewardfile]

    plt.figure()
    plt.hist(rewards,bins=100)
    plt.title(sys.argv[1]+" Reward Histogram")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.savefig(savedir+"/reward-hist.png")
    plt.clf()

    maxrew=max(rewards)


#compute highest reward paths in network
    for source,dest in dg.edges():
        dg.edge[source][dest]['norm_rew']=dg.edge[source][dest]['weight']*(-1*dg.node[dest]['reward']+maxrew)

    path_rewards={}
    for node1 in dg.nodes():
        path_rewards[node1]={}
        for node2 in dg.nodes():
            path_rewards[node1][node2]=0.0

    all_paths=nx.shortest_path(dg,weight='norm_rew')

    with open(savedir+"/best_path_rewards.csv",'w') as out:
        out.write("Source/Dest,"+",".join(dg.nodes())+"\n")
        for node1 in dg.nodes():
            out.write(node1)
            for node2 in all_paths[node1].keys():
                cumulative_reward=0.0
                cumulative_probability=1.0
                traversal=all_paths[node1][node2]
                for sind in range(0,len(traversal)-1):
                    cumulative_probability=cumulative_probability*dg.edge[traversal[sind]][traversal[sind+1]]['weight']
                    cumulative_reward=cumulative_reward+dg.node[traversal[sind+1]]['reward']
                path_rewards[node1][node2]=path_rewards[node1][node2]+cumulative_reward*cumulative_probability
                #print(node1+" "+node2+" "+str(path_rewards[node1][node2]))
            for node2 in dg.nodes():
                out.write(","+str(path_rewards[node1][node2]))
            out.write("\n")
        out.write("\n")

#locate start and end nodes
    source_nodes=[]
    sink_nodes=[]
    for node in dg.nodes():
        if "|" in node:
            continue
        incount=dg.in_degree(node)
        if(incount):
            edgelist=dg.in_edges(node)
            for (src,dest) in edgelist:
                if(node==dest and node==src):
                    incount=incount-1
        if(incount==0):
            source_nodes.append(node)
        outcount=dg.out_degree(node)
        if(outcount):
            edgelist=dg.out_edges(node)
            for (src,dest) in edgelist:
                if(node==dest and node==src):
                    outcount=outcount-1
        if(outcount==0):
            sink_nodes.append(node)

#identify paths between start and end nodes
    for src in source_nodes:
        print("max rewards of initial states to terminal states")
        for dest in sink_nodes:
            print(src+"->"+dest+" "+str(path_rewards[src][dest]))
        print("sum rewards of initial states to terminal states")
        for dest in sink_nodes:
            print(src+"->"+dest+" "+str(distances[src][dest]))


#plot some analytics
    plt.figure()
    plt.scatter([dg.in_degree(node) for node in dg.nodes() if "|" not in node],[dg.node[node]['reward'] for node in dg.nodes() if "|" not in node])
    plt.title(arg+" Indegree vs Reward")
    plt.ylabel("Reward")
    plt.xlabel("In-Degree")
    plt.savefig(savedir+"/indegree-reward.png")
    plt.clf()

    plt.figure()
    plt.scatter([dg.degree(node) for node in dg.nodes() if "|" not in node],[dg.node[node]['reward'] for node in dg.nodes() if "|" not in node])
    plt.title(arg+" Degree vs Reward")
    plt.ylabel("Reward")
    plt.xlabel("Degree")
    plt.savefig(savedir+"/degree-reward.png")
    plt.clf()

    plt.figure()
    plt.scatter([dg.out_degree(node) for node in dg.nodes() if "|" not in node],[dg.node[node]['reward'] for node in dg.nodes() if "|" not in node])
    plt.title(arg+" Outdegree vs Reward")
    plt.ylabel("Reward")
    plt.xlabel("Out-Degree")
    plt.savefig(savedir+"/outdegree-reward.png")
    plt.clf()

    pathvals=[]
    for node1 in source_nodes:
        if "|" in node1:
            continue
        for node2 in sink_nodes:
            if "|" in node2:
                continue
            pathvals.append(distances[node1][node2])


    pathvals=[]
    for node1 in source_nodes:
        if "|" in node1:
            continue
        for node2 in sink_nodes:
            if "|" in node2:
                continue
            pathvals.append(path_rewards[node1][node2])

    plt.figure()
    plt.hist(pathvals)
    plt.title(arg+" Max initial state to end state path reward histogram")
    plt.xlabel("Max reward")
    plt.ylabel("Frequency")
    plt.savefig(savedir+"/max-reward-hist.png")
    plt.clf()

    plt.figure()
    plt.scatter([dg.node[node]['betweenness'] for node in dg.nodes() if "|" not in node],[dg.node[node]['reward'] for node in dg.nodes() if "|" not in node])
    plt.title(arg+" Betweenness vs Reward")
    plt.ylabel("Reward")
    plt.xlabel("Betweenness")
    plt.savefig(savedir+"/betweenness-reward.png")
    plt.clf()

    plt.figure()
    plt.scatter([dg.node[node]['closeness_vitality'] for node in dg.nodes() if "|" not in node],[dg.node[node]['reward'] for node in dg.nodes() if "|" not in node])
    plt.title(arg+" Closeness vitality vs Reward")
    plt.ylabel("Reward")
    plt.xlabel("Closeness vitality")
    plt.savefig(savedir+"/closeness_vitality-reward.png")
    plt.clf()

#read the mapping between the cognitive states and their attribute values
    mapping={}
    with open(basepath+"/save.map") as mapfile:
        for line in mapfile.read().splitlines():
            key,val=line.split("|")
            mapping[key]=val

#compute the rewards for all of the sequences/csv files that were run
    seq_rewards=[]
    seq_sums=[]
    seq_probs=[]
    with open(basepath+"/save.seq") as seqfile:
        for line in seqfile.read().splitlines():
            seqs=line.split(',')[:-1]
            tmp_rew=0
            tmp_act_rew=0
            tmp_prob=1.0
            act_final_rew=0
            prev_act_rew=0
            tmp_act_rew_noncum=0
            for i in range(0,len(seqs),2):
                tmp_rew=tmp_rew+dg.node[seqs[i]]['reward']
            for i in range(0,len(seqs)-2,2):
                tmp_prob=tmp_prob*dg.edge[seqs[i]][seqs[i]+"|"+seqs[i+1]]['weight']
            seq_rewards.append(tmp_prob*tmp_rew)
            seq_sums.append(tmp_rew)
            seq_probs.append(tmp_prob)

    plt.figure()
    plt.hist(seq_sums)
    plt.title(arg+" Sum initial state to end state path reward histogram")
    plt.xlabel("Cumulative reward")
    plt.ylabel("Frequency")
    plt.savefig(savedir+"/path-sum-hist.png")
    plt.clf()

    plt.figure()
    plt.hist(seq_rewards)
    plt.title(arg+" prob*sum initial state to end state path reward histogram")
    plt.xlabel("Cumulative reward*prob")
    plt.ylabel("Frequency")
    plt.savefig(savedir+"/path-reward-hist.png")
    plt.clf()

    plt.figure()
    plt.hist(seq_probs)
    plt.title(arg+" prob initial state to end state path reward histogram")
    plt.xlabel("Cumulative probability")
    plt.ylabel("Frequency")
    plt.savefig(savedir+"/path-prob-hist.png")
    plt.clf()

#    print("rewards learned by irl for actual sequences")
#    print(seq_rewards)
#
#    print("rewards from reward header for actual sequences")
#    print(act_seq_rewards)
#
#    print("Wilcox signed rank statistic for irl path sum rewards vs \"actual\" path sum rewards")
#    print(stats.wilcoxon(seq_rewards,act_seq_rewards))
#
#    print("Wilcox signed rank statistic for irl rewards vs \"actual\" rewards")
#    print(stats.wilcoxon([dg.node[node]['reward'] for node in dg.nodes() if "|" not in node],[dg.node[node]['act_reward'] for node in dg.nodes() if "|" not in node]))

