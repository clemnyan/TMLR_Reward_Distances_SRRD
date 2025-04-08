#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import networkx as nx
import re
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import subprocess
import copy
import os

if not os.path.exists("figs"):
    os.mkdir("figs")


def maxabs(l):
    mx=max([abs(rew) for rew in l])
    return mx

def maxabsnorm(l):
    mx=maxabs(l)
    if mx==0:
        return l
    mxl=[(norm/mx)for norm in l]
    return mxl

resheader="name1,name2,num samples,unnorm wilcox s,unnorm wilcox p,norm wilcox s,norm wilcox p,norm binom. p,spear r,spear p,ttest stat,ttest p"
def printstats(name1,name2,l1,l2):
    if len(l1)<2:
        return
    norm_l1=maxabsnorm(l1)
    norm_l2=maxabsnorm(l2)
    wnorm=stats.wilcoxon(norm_l1,norm_l2)
    wunnorm=stats.wilcoxon(l1,l2)
    bino_p=stats.binom_test(sum([(a-b)>0 for a,b in zip(l1,l2)]),n=len(l1),p=0.5,alternative='two-sided')
    #spear=stats.spearmanr(norm_l1,norm_l2)
    spear=stats.spearmanr(l1,l2)
    tval=stats.ttest_rel(norm_l1,norm_l2)
    vals=[name1,name2,len(l1),wunnorm.statistic,wunnorm.pvalue,wnorm.statistic,wnorm.pvalue,bino_p,spear.correlation,spear.pvalue,tval.statistic,tval.pvalue]
    print(','.join(map(str,vals)))
    plt.figure()
    plt.scatter(l1,l2)
    plt.title(name1+" vs. "+name2)
    plt.ylabel(name2)
    plt.xlabel(name1)
    plt.savefig("figs/"+name1[:name1.find("/")]+"-"+name2[:name2.find("/")]+".png")
    plt.clf()
    plt.close()
    plt.figure()
    plt.scatter(norm_l1,norm_l2)
    plt.title(name1+" vs. "+name2+" (normalized)")
    plt.ylabel(name2)
    plt.xlabel(name1)
    plt.savefig("figs/"+name1[:name1.find("/")]+name2[:name2.find("/")]+"-normalized.png")
    plt.clf()
    plt.close()

arguments=sys.argv[1:]
edgepattern='"([A-Z]+)"->"([A-Z]+)"\[label="(.*)\|.*\|([^"]+)".*'
#edgepattern='"([A-Z]+)"->"([A-Z]+)".*\|.*\|([^"]+)".*'
nodepattern='"([A-Z]+)"\[.*\|([^"]+).*'
nodepatternnoreward='"([A-Z]+)"\[.*'

edgere=re.compile(edgepattern)
nodere=re.compile(nodepattern)
nodenorewardre=re.compile(nodepatternnoreward)

dgs={}
mapping={}
seq_rewards={}
seq_sums={}
act_seq_sums={}
act_seq_rewards={}
act_seq_rewards_noncum={}
act_final_state_rewards={}
tmp_rewards_debug={}
seq_probs={}
basepaths={}
filenames={}
training_files={}
seq_files={}
map_files={}
action_files={}
attribute_files={}
dot_files={}
dtm_files={}
for arg in arguments:
    basepath="./"
    if arg.rfind("/")>0:
        basepath=arg[:arg.rfind("/")+1]

    has_act=False
    if not os.path.exists(arg):
        continue

    filename=arg
    actname=""

    if os.path.isdir(arg):
        filename=arg+"/save1.dot"
        basepath=arg+"/"

    basepaths[arg]=basepath
    filenames[arg]=filename
    
    action_files[arg]=basepath+"actions.csv"
    attribute_files[arg]=basepath+"attributes.csv"
    training_files[arg]=re.sub("-c100-d0.5-l0.5-e5000.dtm.dot",".files",basepath+"list.files")
    seq_files[arg]=re.sub(".dtm.dot",".seq",filename)
    seq_files[arg]=re.sub("1.dot",".seq",filename)
    map_files[arg]=re.sub(".dtm.dot",".map",filename)
    map_files[arg]=re.sub("1.dot",".map",filename)
    dot_files[arg]=filename
    dtm_files[arg]=re.sub("1.dot",".1",filename)

    #generate the .map and .seq files because they may not exist
    #print("./DTM_makeSeqMap -dtm "+dtm_files[arg]+" -training "+training_files[arg]+" -attributes "+attribute_files[arg]+" -actions "+action_files[arg]+" -seq "+seq_files[arg]+" -map "+map_files[arg]+" -measure vector")
    subprocess.call("./DTM_makeSeqMap -dtm "+dtm_files[arg]+" -training "+training_files[arg]+" -attributes "+attribute_files[arg]+" -actions "+action_files[arg]+" -seq "+seq_files[arg]+" -map "+map_files[arg]+" -measure vector",shell=True,stdout=None)

    #read .dot file and make a graph called dg
    #with open("test.dg",mode='r') as gvfile:
    dg=nx.DiGraph()
    linenum=0
    tmp_rewards_debug[arg]=[]
    with open(dot_files[arg],mode='r') as gvfile:
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
                tmp_rewards_debug[arg].append(reward)
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
                edgematch=edgere.search(line)
                if nodematch and not edgematch:
                    node=nodematch.group(1)
                    reward=float(nodematch.group(2))
                    if(node in dg):
                        dg.node[node]['act_reward']=reward
    #make variables so that when this is run inside of python and not from the
    #script that the variables stay in the environment
    betweenness=nx.betweenness_centrality(dg)
    close_vitality=nx.closeness_vitality(dg,weight='weight')
    for b in betweenness:
        dg.node[b]['betweenness']=betweenness[b]
        dg.node[b]['closeness_vitality']=close_vitality[b]
    del close_vitality
    del betweenness

    #initialize all nodes that don't have a reward
    for node in dg.nodes():
        if "reward" not in dg.node[node].keys():
            dg.node[node]['reward']=0.0
        if "act_reward" not in dg.node[node].keys():
            dg.node[node]['act_reward']=0.0


    #read the mapping between the cognitive states and their attribute values
    mapping[arg]={}
    #print(map_files[arg])
    with open(map_files[arg]) as mapfile:
        for line in mapfile.read().splitlines():
            key,val=line.split("|")
            mapping[arg][key]=val

    #compute the rewards for all of the sequences/csv files that were run
    seq_rewards[arg]=[]
    seq_sums[arg]=[]
    act_seq_sums[arg]=[]
    act_seq_rewards[arg]=[]
    act_seq_rewards_noncum[arg]=[]
    act_final_state_rewards[arg]=[]
    seq_probs[arg]=[]
    #with open(sys.argv[1]+"/save.seq") as seqfile:
    #print(seq_files[arg])
    with open(seq_files[arg]) as seqfile:
        for line in seqfile.read().splitlines():
            seqs=line.split('|')[:-1]
            tmp_rew=0
            tmp_act_rew=0
            tmp_prob=1.0
            act_final_rew=0
            prev_act_rew=0
            tmp_act_rew_noncum=0
            for i in range(0,len(seqs),2):
                tmp_rew=tmp_rew+dg.node[seqs[i]]['reward']
                if has_act:
                    tmp_act_rew=tmp_act_rew+dg.node[seqs[i]]['act_reward']
                    tmp_act_rew_noncum=tmp_act_rew_noncum+dg.node[seqs[i]]['act_reward']-prev_act_rew
                    prev_act_rew=dg.node[seqs[i]]['act_reward']
                    act_final_rew=dg.node[seqs[i]]['act_reward']
            for i in range(0,len(seqs)-2,2):
                tmp_prob=tmp_prob*dg.edge[seqs[i]][seqs[i]+"|"+seqs[i+1]]['weight']
            seq_rewards[arg].append(tmp_prob*tmp_rew)
            seq_sums[arg].append(tmp_rew)
            seq_probs[arg].append(tmp_prob)
            if has_act:
                act_seq_sums[arg].append(tmp_act_rew)
                act_seq_rewards[arg].append(tmp_prob*tmp_act_rew)
                act_seq_rewards_noncum[arg].append(tmp_prob*tmp_act_rew_noncum)
                act_final_state_rewards[arg].append(act_final_rew)
    dgs[arg]=copy.deepcopy(dg)

#print(sorted([dgs['loser'].node[node]['reward'] for node in dgs['loser'].nodes()]))
#print(sorted(tmp_rewards_debug['loser']))
both_map={}
#print(sys.argv[1:])
for arg in arguments:
    if arg not in both_map.keys():
        both_map[arg]={}
    for arg2 in arguments:
        if arg==arg2:
            continue
        if arg2 not in both_map[arg].keys():
            both_map[arg][arg2]={}
        for node in dgs[arg].nodes():
            nodepart=node.split("|")[0]
            actionpart=None
            if "|" in node:
                actionpart=node.split("|")[1]
            for k in dgs[arg2].nodes():
                nodekpart=k.split("|")[0]
                actionkpart=None
                if "|" in k:
                    actionkpart=k.split("|")[1]
                #this is an edge node in one and not in the other, no mapping
                if (actionpart and not actionkpart) or (actionkpart and not actionpart):
                    continue
                elif(actionpart and actionkpart):
                    #need to match both action and node
                    if actionpart == actionkpart and mapping[arg2][nodekpart]==mapping[arg][nodepart]:
                        both_map[arg][arg2][node]=k
                    pass
                else:
                    if mapping[arg2][k]==mapping[arg][node]:
                        both_map[arg][arg2][node]=k

print(resheader)
preds={}
for arg in arguments:
    preds[arg]={}
    for arg2 in arguments:
        if arg2 == arg:
            continue
        preds[arg][arg2]=[]

for arg in arguments:
    for arg2 in arguments:
        if arg2==arg:
            continue
        #print(training_files[arg2])
        with open(training_files[arg2],"r") as lfile:
            for line in lfile.read().splitlines():
                if(line is None or line.strip() == ''):
                    continue
                #print("./DTM_scenarioMatch -dtm "+dtm_files[arg]+" -attributes "+attribute_files[arg]+" -actions "+action_files[arg]+" -measure vector -scenario "+line)
                proc=subprocess.Popen("./DTM_scenarioMatch -dtm "+dtm_files[arg]+" -attributes "+attribute_files[arg]+" -actions "+action_files[arg]+" -measure vector -scenario "+line,shell=True,stdout=subprocess.PIPE)
                output=proc.communicate()
                print(output)
                val=float(output[0].decode('utf8').strip().splitlines()[-1].split('\t')[0])
                #print(output[0].decode('utf8').strip().splitlines()[-1].split('\t'))
                #print(val)
                #val=float(output[0].splitlines()[-1].split('\t')[0])
                preds[arg][arg2].append(val)
        #print(arg+" "+arg2+" "+",".join([str(pred) for pred in preds[arg][arg2]]))
        #print(arg2+" "+",".join([str(se) for se in seq_rewards[arg]]))
        
for arg in arguments:
    #print("# cog states in "+arg+":"+str(len(dgs[arg].nodes())))
    for arg2 in arguments:
        if arg2==arg:
            continue
        #print("# cog states in union("+arg+","+arg2+"):"+str(len(both_map[arg][arg2])))
        #print("% cog states in union("+arg+","+arg2+"):"+str(100.0*float(len(both_map[arg][arg2]))/float(len(dgs[arg].nodes()))))
        orig_toy_rew=[]
        orig_seq_rew=[]
        orig_seq_act_rew=[]
        orig_seq_rew_sum=[]
        other_seq_rew=[]
        other_seq_act_rew=[]
        other_seq_rew_sum=[]
        seq_lens=[]
        act_rew_sum_noncum=[]
        act_rew_prod_noncum=[]
        with open(seq_files[arg]) as seqfile:
            for line in seqfile.read().splitlines():
                seqs=line.split(',')[:-1]
                found=True
                for node in seqs:
                    if node not in both_map[arg][arg2].keys():
                        found=False
                        break
                if not found:
                    continue
                tmp_rew1=0
                tmp_act_rew1=0
                tmp_prob1=1.0
                tmp_rew2=0
                tmp_act_rew2=0
                tmp_prob2=1.0
                seq_lens.append(len(seqs))
                prev_act_rew1=0
                tmp_act_sum=0
                for i in range(0,len(seqs)):
                    tmp_rew1=tmp_rew1+dgs[arg].node[seqs[i]]['reward']
                    if has_act:
                        tmp_act_rew1=tmp_act_rew1+dgs[arg].node[seqs[i]]['act_reward']
                        tmp_act_sum=tmp_act_sum+dgs[arg].node[seqs[i]]['act_reward']-prev_act_rew1
                        prev_act_rew1=dgs[arg].node[seqs[i]]['act_reward']
                        tmp_act_rew2=tmp_act_rew2+dgs[arg2].node[both_map[arg][arg2][seqs[i]]]['act_reward']
                    #print(dgs[arg].node[seqs[i]]['act_reward'])
                    #print(prev_act_rew1)
                    tmp_rew2=tmp_rew2+dgs[arg2].node[both_map[arg][arg2][seqs[i]]]['reward']
                for i in range(1,len(seqs)-1):
                    tmp_prob1=tmp_prob1*dgs[arg].edge[seqs[i]][seqs[i+1]]['weight']
                    other_src=both_map[arg][arg2][seqs[i]]
                    other_dest=both_map[arg][arg2][seqs[i+1]]
                    tmp_prob2=tmp_prob2*dgs[arg2].edge[other_src][other_dest]['weight']
                if has_act:
                    act_rew_sum_noncum.append(tmp_act_sum)
                    act_rew_prod_noncum.append(tmp_act_sum*tmp_prob1)
                    orig_seq_act_rew.append(tmp_prob1*tmp_act_rew1)
                    orig_toy_rew.append(tmp_act_rew1)
                    other_seq_act_rew.append(tmp_prob2*tmp_act_rew2)
                orig_seq_rew.append(tmp_prob1*tmp_rew1)
                orig_seq_rew_sum.append(tmp_rew1)
                other_seq_rew.append(tmp_prob2*tmp_rew2)
                other_seq_rew_sum.append(tmp_rew2)

        orignodes=[]
        othernodes=[]
        for node in dgs[arg].nodes():
            if node in both_map[arg][arg2].keys() and "|" not in node:
                orignodes.append(node)
                othernodes.append(both_map[arg][arg2][node])

        #irl learned reward vs irl learned reward for other actor at the cognitive state level
        printstats(arg+" cs reward",arg2+" cs reward",[dgs[arg].node[node]['reward'] for node in orignodes if "|" not in node],[dgs[arg2].node[node]['reward'] for node in othernodes if "|" not in node])
        #irl learned reward sum (no prob) vs game reward sum (no prob)
        if has_act:
            printstats(arg+" path sum reward","toy path sum reward",orig_seq_rew_sum,orig_toy_rew)
        #irl learned reward sum (no prob) vs game reward sum noncumulative (no prob)
        if has_act:
            printstats(arg+" path sum reward","toy path sum reward (noncum)",orig_seq_rew_sum,act_rew_sum_noncum)
        #irl learned reward sum (no prob)/len vs game reward sum (no prob)/len
        #printstats(arg+" path sum reward/len","toy path sum reward/len",[orig_seq_rew_sum[i]/seq_lens[i] for i in range(0,len(orig_seq_rew_sum))],[orig_toy_rew[i]/seq_lens[i] for i in range(0,len(orig_toy_rew))])
        #irl learned reward prod vs game reward prod
        if has_act:
            printstats(arg+" path prob x reward","toy path prob x reward",orig_seq_rew,orig_seq_act_rew)
        #irl learned reward prod vs game reward prod (noncum)
        if has_act:
            printstats(arg+" path prob x reward","toy path prob x reward (noncum)",orig_seq_rew,act_rew_prod_noncum)
        #irl learned reward sum (no prob) vs irl learned reward sum (no prob)
        printstats(arg+" sum path reward",arg2+" sum path reward",orig_seq_rew_sum,other_seq_rew_sum)
        #irl learned reward prod vs irl learned reward prod
        printstats(arg+" sum path prob x reward",arg2+" sum path prob x reward",orig_seq_rew,other_seq_rew)
        #irl learned reward prod vs predicted reward
        printstats(arg+" path prob x reward",arg2+" pred "+arg+" path prob x reward",seq_rewards[arg],preds[arg2][arg])
        #irl learned reward sum vs predicted reward sum
        printstats(arg+" path reward sum",arg2+" pred "+arg+" path reward sum",seq_sums[arg],preds[arg2][arg])
        #irl learned reward prod vs toy reward prod (all values, not intersection of arg and arg2
        if has_act:
            printstats(arg+" path prob x reward (full)","toy path prob x reward (full)",seq_rewards[arg],act_seq_rewards[arg])
        #irl learned reward sum vs toy reward sum (all values, not intersection of arg and arg2
        if has_act:
            printstats(arg+" path sum reward (full)","toy path sum reward (full)",seq_sums[arg],act_seq_sums[arg])
        #irl learned reward prod vs toy reward prod (all values, not intersection of arg and arg2
        if has_act:
            printstats(arg+" path prob x reward (full)","toy path prob x reward (full-noncum)",seq_rewards[arg],act_seq_rewards_noncum[arg])
        #irl learned reward prod vs toy reward prod (all values, not intersection of arg and arg2
        if has_act:
            printstats(arg+" path prob x reward (full)","toy path final state reward(full)",seq_rewards[arg],act_final_state_rewards[arg])



#for i in `cat learner/list.files`;do ./DTM_scenarioMatch -dtm learner/save.1 -scenario $i -attributes learner/attributes.csv -actions learner/actions.csv -measure vector |tail -n1 >> new;./DTM_scenarioMatch.old -dtm learner/save.1 -scenario $i -attributes learner/attributes.csv -actions learner/actions.csv|tail -n1 >> old;done
