#!/usr/bin/env python3

import networkx as nx
import re
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import subprocess
import copy

def maxabs(l):
    mx=max([abs(rew) for rew in l])
    return mx

def maxabsnorm(l):
    mxl=[(norm/maxabs(l)) for norm in l]
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
    plt.savefig("figs/"+name1+"-"+name2+".png")
    plt.clf()
    plt.figure()
    plt.scatter(norm_l1,norm_l2)
    plt.title(name1+" vs. "+name2+" (normalized)")
    plt.ylabel(name2)
    plt.xlabel(name1)
    plt.savefig("figs/"+name1+"-"+name2+"-normalized.png")
    plt.clf()

arguments=sys.argv[1:]

edgepattern='"([A-Z]+)"->"([A-Z]+)".*\|.*\|([^"]+)".*'
nodepattern='"([A-Z]+)"\[.*\|([^"]+).*'
nodepatternnoreward='"([A-Z]+)"\[.*'

edgere=re.compile(edgepattern)
nodere=re.compile(nodepattern)
nodenorewardre=re.compile(nodepatternnoreward)

dgs={}
mapping={}
seq_rewards={}
act_seq_rewards={}
act_seq_rewards_noncum={}
act_final_state_rewards={}
for arg in arguments:
    #read .dot file and make a graph called dg
    #with open("test.dg",mode='r') as gvfile:
    dg=nx.DiGraph()
    linenum=0
    with open(arg+"/save1.dot",mode='r') as gvfile:
        for line in gvfile.readlines():
            linenum=linenum+1
            edgematch=edgere.search(line)
            nodematch=nodere.search(line)
            nodenorewardmatch=nodenorewardre.search(line)
            if(edgematch):
                node1=edgematch.group(1)
                node2=edgematch.group(2)
                prob=float(edgematch.group(3))
                #print("matched an edge "+node1+" "+node2+" "+str(prob))
                if(node1 in dg and node2 in dg):
                    #print("added edge")
                    dg.add_edge(node1,node2,weight=prob)
                else:
                    if(node1 in dg):
                        print(node2+" not found on line "+str(linenum))
                    else:
                        print(node1+" not found on line "+str(linenum))
            elif(nodematch):
                node=nodematch.group(1)
                reward=float(nodematch.group(2))
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

    #read the mapping between the cognitive states and their attribute values
    mapping[arg]={}
    with open(arg+"/save.map") as mapfile:
        for line in mapfile.read().splitlines():
            key,val=line.split(";")
            mapping[arg][key]=val

    #compute the rewards for all of the sequences/csv files that were run
    seq_rewards[arg]=[]
    act_seq_rewards[arg]=[]
    act_seq_rewards_noncum[arg]=[]
    act_final_state_rewards[arg]=[]
    #with open(sys.argv[1]+"/save.seq") as seqfile:
    with open(arg+"/save.seq") as seqfile:
        for line in seqfile.read().splitlines():
            seqs=line.split(',')[:-1]
            tmp_rew=0
            tmp_act_rew=0
            tmp_prob=1.0
            act_final_rew=0
            prev_act_rew=0
            tmp_act_rew_noncum=0
            for i in range(1,len(seqs)):
                tmp_rew=tmp_rew+dg.node[seqs[i]]['reward']
                tmp_act_rew=tmp_act_rew+dg.node[seqs[i]]['act_reward']
                tmp_act_rew_noncum=tmp_act_rew_noncum+dg.node[seqs[i]]['act_reward']-prev_act_rew
                prev_act_rew=dg.node[seqs[i]]['act_reward']
                act_final_rew=dg.node[seqs[i]]['act_reward']
            for i in range(0,len(seqs)-1):
                tmp_prob=tmp_prob*dg.edge[seqs[i]][seqs[i+1]]['weight']
            seq_rewards[arg].append(tmp_prob*tmp_rew)
            act_seq_rewards[arg].append(tmp_prob*tmp_act_rew)
            act_seq_rewards_noncum[arg].append(tmp_prob*tmp_act_rew_noncum)
            act_final_state_rewards[arg].append(act_final_rew)
    dgs[arg]=copy.deepcopy(dg)


both_map={}
#print(sys.argv[1:])
for arg in arguments:
    if arg not in both_map.keys():
        both_map[arg]={}
    for arg2 in arguments:
        #if arg==arg2:
        #    continue
        if arg2 not in both_map[arg].keys():
            both_map[arg][arg2]={}
        for node in dgs[arg].nodes():
            for k in dgs[arg2].nodes():
                if mapping[arg2][k]==mapping[arg][node]:
                    both_map[arg][arg2][node]=k

print(resheader)
preds={}
for arg in arguments:
    preds[arg]={}
    for arg2 in arguments:
        #if arg2 == arg:
        #    continue
        preds[arg][arg2]=[]

for arg in arguments:
    for arg2 in arguments:
        #if arg2==arg:
        #    continue
        with open(arg2+"/list.files","r") as lfile:
            for line in lfile.read().splitlines():
                proc=subprocess.Popen("./DTM_scenarioMatch -dtm "+arg+"/save.1 -attributes "+arg+"/attributes.csv -actions "+arg+"/actions.csv -scenario "+line,shell=True,stdout=subprocess.PIPE)
                output=proc.communicate()
                val=float(output[0].decode('utf8').strip().splitlines()[-1].split('\t')[0])
                print(val)
                #val=float(output[0].splitlines()[-1].split('\t')[0])
                preds[arg][arg2].append(val)
        print(arg+" "+arg2+" "+",".join([str(pred) for pred in preds[arg][arg2]]))
        print(arg2+" "+",".join([str(se) for se in seq_rewards[arg]]))
        
sys.exit(0)
for arg in arguments:
    print("# cog states in "+arg+":"+str(len(dgs[arg].nodes())))
    for arg2 in arguments:
        #if arg2==arg:
        #    continue
        print("# cog states in union("+arg+","+arg2+"):"+str(len(both_map[arg][arg2])))
        print("% cog states in union("+arg+","+arg2+"):"+str(100.0*float(len(both_map[arg][arg2]))/float(len(dgs[arg].nodes()))))
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
        with open(arg+"/save.seq") as seqfile:
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
                    tmp_act_rew1=tmp_act_rew1+dgs[arg].node[seqs[i]]['act_reward']
                    tmp_act_sum=tmp_act_sum+dgs[arg].node[seqs[i]]['act_reward']-prev_act_rew1
                    #print(dgs[arg].node[seqs[i]]['act_reward'])
                    #print(prev_act_rew1)
                    prev_act_rew1=dgs[arg].node[seqs[i]]['act_reward']
                    tmp_rew2=tmp_rew2+dgs[arg2].node[both_map[arg][arg2][seqs[i]]]['reward']
                    tmp_act_rew2=tmp_act_rew2+dgs[arg2].node[both_map[arg][arg2][seqs[i]]]['act_reward']
                for i in range(1,len(seqs)-1):
                    tmp_prob1=tmp_prob1*dgs[arg].edge[seqs[i]][seqs[i+1]]['weight']
                    other_src=both_map[arg][arg2][seqs[i]]
                    other_dest=both_map[arg][arg2][seqs[i+1]]
                    tmp_prob2=tmp_prob2*dgs[arg2].edge[other_src][other_dest]['weight']
                act_rew_sum_noncum.append(tmp_act_sum)
                act_rew_prod_noncum.append(tmp_act_sum*tmp_prob1)
                orig_seq_rew.append(tmp_prob1*tmp_rew1)
                orig_seq_rew_sum.append(tmp_rew1)
                orig_seq_act_rew.append(tmp_prob1*tmp_act_rew1)
                orig_toy_rew.append(tmp_act_rew1)
                other_seq_rew.append(tmp_prob2*tmp_rew2)
                other_seq_rew_sum.append(tmp_rew2)
                other_seq_act_rew.append(tmp_prob2*tmp_act_rew2)

        orignodes=[]
        othernodes=[]
        for node in dgs[arg].nodes():
            if node in both_map[arg][arg2].keys():
                orignodes.append(node)
                othernodes.append(both_map[arg][arg2][node])

        #irl learned reward vs irl learned reward for other actor at the cognitive state level
        printstats(arg+" cs reward",arg2+" cs reward",[dgs[arg].node[node]['reward'] for node in orignodes],[dgs[arg2].node[node]['reward'] for node in othernodes])
        #irl learned reward sum (no prob) vs game reward sum (no prob)
        printstats(arg+" path sum reward","toy path sum reward",orig_seq_rew_sum,orig_toy_rew)
        #irl learned reward sum (no prob) vs game reward sum noncumulative (no prob)
        printstats(arg+" path sum reward","toy path sum reward (noncum)",orig_seq_rew_sum,act_rew_sum_noncum)
        #irl learned reward sum (no prob)/len vs game reward sum (no prob)/len
        #printstats(arg+" path sum reward/len","toy path sum reward/len",[orig_seq_rew_sum[i]/seq_lens[i] for i in range(0,len(orig_seq_rew_sum))],[orig_toy_rew[i]/seq_lens[i] for i in range(0,len(orig_toy_rew))])
        #irl learned reward prod vs game reward prod
        printstats(arg+" path prob x reward","toy path prob x reward",orig_seq_rew,orig_seq_act_rew)
        #irl learned reward prod vs game reward prod (noncum)
        printstats(arg+" path prob x reward","toy path prob x reward (noncum)",orig_seq_rew,act_rew_prod_noncum)
        #irl learned reward sum (no prob) vs irl learned reward sum (no prob)
        printstats(arg+" sum path reward",arg2+" sum path reward",orig_seq_rew_sum,other_seq_rew_sum)
        #irl learned reward prod vs irl learned reward prod
        printstats(arg+" sum path prob x reward",arg2+" sum path prob x reward",orig_seq_rew,other_seq_rew)
        #irl learned reward prod vs predicted reward
        printstats(arg+" path prob x reward",arg2+" pred "+arg+" path prob x reward",seq_rewards[arg],preds[arg2][arg])
        #irl learned reward prod vs toy reward prod (all values, not intersection of arg and arg2
        printstats(arg+" path prob x reward (full)","toy path prob x reward (full)",seq_rewards[arg],act_seq_rewards[arg])
        #irl learned reward prod vs toy reward prod (all values, not intersection of arg and arg2
        printstats(arg+" path prob x reward (full)","toy path prob x reward (full-noncum)",seq_rewards[arg],act_seq_rewards_noncum[arg])
        #irl learned reward prod vs toy reward prod (all values, not intersection of arg and arg2
        printstats(arg+" path prob x reward (full)","toy path final state reward(full)",seq_rewards[arg],act_final_state_rewards[arg])



#for i in `cat learner/list.files`;do ./DTM_scenarioMatch -dtm learner/save.1 -scenario $i -attributes learner/attributes.csv -actions learner/actions.csv -measure vector |tail -n1 >> new;./DTM_scenarioMatch.old -dtm learner/save.1 -scenario $i -attributes learner/attributes.csv -actions learner/actions.csv|tail -n1 >> old;done
