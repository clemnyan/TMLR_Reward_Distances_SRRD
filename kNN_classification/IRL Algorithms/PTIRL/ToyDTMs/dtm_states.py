#
# Name:		dtm_states.py
# Date:		2018-01-03
# Project:	DTM
# Copyright Eugene Santos Jr. (2018)
#
import pydtm
import math
import heapq
import datetime
import copy
import dtm_neighborhood
import dtm_irl

def DTM_stateDistance(lespace, attr1, target_attr1, attr2, target_attr2, hamming = False): # attr1 and target_attr1 are the disctionaries capturing the single learning episode in a state (assumes we only have one learning episode per state)
	# Note -- if attribute values are not numeric, then the distance is simply 1 if they are not the same.
	# 		If attribute value not found for one or the other, or there is a type difference, the distance for those attributes are 1
	# hamming = True implies that no distance is computed between numerical cvalues.

#	print ('attr1 = ', end='')
#	print (attr1)
#	print ('attr2 = ', end='')
#	print (attr2)
	distance = 0.0
	for key1, value1 in attr1.items():
		try:
			if hamming == True:
				raise ValueError
			val1 = float(lespace.attribute_vals[int(key1)][int(value1)])
			try:
				value2 = attr2[key1]
				try:
					val2 = float(lespace.attribute_vals[int(key1)][int(value2)])
					distance += (val1 - val2) * (val1 - val2)
				except ValueError:
					distance += 1.0
			except KeyError:
				distance += 1.0
		except ValueError:
			try:
				value2 = attr2[key1]
				if value1 != value2:
					distance += 1.0
			except KeyError:
				distance += 1.0
	for key1, value1 in target_attr1.items():
		try:
			if hamming == True:
				raise ValueError
			val1 = float(lespace.attribute_vals[int(key1)][int(value1)])
			try:
				value2 = target_attr2[key1]
				try:
					val2 = float(lespace.attribute_vals[int(key1)][int(value2)])
					distance += (val1 - val2) * (val1 - val2)
				except ValueError:
					distance += 1.0
			except KeyError:
				distance += 1.0
		except ValueError:
			try:
				value2 = target_attr2[key1]
				if value1 != value2:
					distance += 1.0
			except KeyError:
				distance += 1.0
				
	for key2, value2 in attr2.items():
		try:
			attr1[key2]
		except KeyError:
			distance += 1.0
				
	for key2, value2 in target_attr2.items():
		try:
			target_attr2[key2]
		except KeyError:
			distance += 1.0
	return (math.sqrt(distance))


def DTM_allPairsShortestPath(graph, num_states, num_actions): # Floyd-Warshall
	# Returns dist[sidx1][sidx2]
	dist = {}
	_dist = {}
	_rev_dist = {}
	for sidx1 in range(num_states):
		dist[sidx1] = {}
		_dist[sidx1] = set()
		for sidx2 in range(num_states):
			try:
				_rev_dist[sidx2]
			except KeyError:
				_rev_dist[sidx2] = set()
			if sidx1 == sidx2:
				dist[sidx1][sidx2] = 0
			elif len(graph[2][sidx1][sidx2]) > 0:
				dist[sidx1][sidx2] = 1
				_dist[sidx1].add(sidx2)
				_rev_dist[sidx2].add(sidx1)

	for sidx1 in range(num_states):
		for sidx2 in range(num_states):
			for sidx3 in _dist[sidx1] & _rev_dist[sidx2]:
				try:
					value = dist[sidx1][sidx2]
					if value > dist[sidx1][sidx3] + dist[sidx3][sidx2]:
						dist[sidx1][sidx2] = dist[sidx1][sidx3] + dist[sidx3][sidx2]
				except KeyError:
					try:
						dist[sidx1][sidx2] = dist[sidx1][sidx3] + dist[sidx3][sidx2]
						_dist[sidx1].add(sidx2)
						_rev_dist[sidx2].add(sidx1)
					except KeyError:
						pass

	return (dist)
	

def DTM_computeTrajectoryDistance(traj1, traj2, dist):
	# Distance is determined by the two "closest" states between traj1 and traj2
	#	Hence, if traj1 and traj2 share at least one state, the distance is 0
	traj_dist = None
	for idx1 in range(len(traj1)):
		if idx1 % 2 == 1: # action
			continue
		for idx2 in range(len(traj2)):
			if idx2 % 2 == 1: # action
				continue
			try:
				if traj_dist == None:
					traj_dist = dist[traj1[idx1]][traj2[idx2]]
				elif traj_dist > dist[traj1[idx1]][traj2[idx2]]:
					traj_dist = dist[traj1[idx1]][traj2[idx2]]
			except KeyError:
				continue
	return (traj_dist)


def commonPrefix(l1, l2):
	for i, c in enumerate(l1):
		if c != l2[i]:
			return (i)
	return (i)


def DTM_computeMergeStates(dtm, graph, trajectories, ignore_start_prefix = True, percent_distant_pairs = 10.0, merges_per_pair = 5):
	# This function proposes states to unify in order to guarantee overlap
	#	the trajectories. 
	# ignore_start_prefix means that if two trajectories share a common prefix
	#	of states (ignoring actions) that the common prefix is ignored
	#	in the distance computation between trajectories. This could
	#	also result in entire trajectories being a prefix or identical
	#	which implies a 0 distance.
	# percent_distant_pairs is the % of pairs in the non-zero trajectory
	#	distance list that should be resolved.
	# merges_per_pair is the number of non-zero states to merge if any.
	#	merges_per_pair == 0 implies at least one merge if there are no
	#	overlapping states in the pair.
	# Returns a new dtm and new graph with new dtm triples corresponding to merged states

	# Examine the graph
	print (datetime.datetime.now())
	print ('Computing all pairs shortest distance between states---')
	dist = DTM_allPairsShortestPath(graph, dtm.num_cs, dtm.num_a)
	print (datetime.datetime.now())

	# Now for all pairs of trajectories, determine the distance between them
	#	by computing the shortest distance between any two states
	print ('Computing inter-trajectory distances---')
	h = list()
	traj_dist = {}
	non_zero_count = 0
	for tidx1 in range(len(trajectories) - 1):
		traj_dist[tidx1] = {}
		for tidx2 in range(tidx1 + 1, len(trajectories)):
			if ignore_start_prefix == True:
				i = commonPrefix(trajectories[tidx1], trajectories[tidx2])
				if i % 2 == 1:
					i += 1
				traj_dist[tidx1][tidx2] = DTM_computeTrajectoryDistance(trajectories[tidx1][i:], trajectories[tidx2][i:], dist)
			else:
				i = 0
				traj_dist[tidx1][tidx2] = DTM_computeTrajectoryDistance(trajectories[tidx1], trajectories[tidx2], dist)
			try:
#				if traj_dist[tidx1][tidx2] > 0: # only heap non-zero distanaces
#					heapq.heappush(h, ( -traj_dist[tidx1][tidx2], tidx1, tidx2, i ) )
				heapq.heappush(h, ( -traj_dist[tidx1][tidx2], tidx1, tidx2, i ) ) # heap zero distances also
				if traj_dist[tidx1][tidx2] > 0: 
					non_zero_count += 1
			except TypeError: # Trajectory distance was None!
				heapq.heappush(h, ( -10*dtm.num_cs, tidx1, tidx2, i ) )
	print ('Total non-zero distance trajectory pairs = ', end='')
	print (len(h))
	print (datetime.datetime.now())

	print ('Building hash tables for state to trajectories mappings---')
	state_to_trajs = {}
	for tidx, traj in enumerate(trajectories):
		for pos, sidx in enumerate(traj):
			if pos % 2 == 0: # state
				try:
					state_to_trajs[sidx].add(tidx)
				except KeyError:
					state_to_trajs[sidx] = set()
					state_to_trajs[sidx].add(tidx)
	print (datetime.datetime.now())

	print ('Copying dtm ...')
	dtm = copy.deepcopy(dtm)
	print (datetime.datetime.now())
	print ('Making new graph...')
	graph = dtm_neighborhood.DTM_makeGraph(dtm)
	print (datetime.datetime.now())

	if merges_per_pair > 0:
		percent_distant_pairs = 100
	print ('Reducing by order of distance trajectory pairs by at least ' + str(percent_distant_pairs) + '% of them')
	reduction = max(int(float(non_zero_count) * float(percent_distant_pairs) / 100.0), 1)
	remainder = max(len(h) - reduction, 0)
	s_dist = {}
	choices = set()
	while len(h) > remainder: # Keep looping
		t_dist, tidx1, tidx2, i = heapq.heappop(h)
		t_dist *= -1
		print ('\tReducing Trajectory #' + str(tidx1+1) + ' and Trajectory #' + str(tidx2+1) + '\tdistance = ' + str(t_dist) + '\tprefix = ' + str(i))

		# build heap of state indices
		s_heap = list()
		match = False
		for pos1, sidx1 in enumerate(trajectories[tidx1][i:]):
			if pos1 % 2 == 1: # action
				continue
			s1 = dtm.states[sidx1]
			s1key = s1.key
			le_idx1 = int(s1key[:-1])
			le1 = dtm.lespace.les[le_idx1]
			for pos2, sidx2 in enumerate(trajectories[tidx2][i:]):
				if sidx1 == sidx2: # skip
					continue
				if pos2 % 2 == 1: # action
					continue
				try:
					d = s_dist[sidx1][sidx2]
				except KeyError:
					s2 = dtm.states[sidx2]
					s2key = s2.key
					le_idx2 = int(s2key[:-1])
					le2 = dtm.lespace.les[le_idx2]
					d = DTM_stateDistance(dtm.lespace, le1.attributes, le1.target_attributes, le2.attributes, le2.target_attributes)
					try:
						s_dist[sidx1][sidx2] = d
					except KeyError:
						s_dist[sidx1] = {}
						s_dist[sidx1][sidx2] = d
					try:
						s_dist[sidx2][sidx1] = d
					except KeyError:
						s_dist[sidx2] = {}
						s_dist[sidx2][sidx1] = d
				if d == 0: # skip
					match = True
					continue
				if ( d, sidx1, sidx2 ) in s_heap or ( d, sidx2, sidx1 ) in s_heap:
					match = True
					continue
				heapq.heappush(s_heap, ( d, sidx1, sidx2 ) )
		# Select states to merge	
		midx = merges_per_pair # positive means have to merge non-zeros regardless of a match
		if match == False: # merge at least one if possible
			midx = max(1, merges_per_pair)
		while midx > 0: # try to get up to merges_per_pair merges
			if len(s_heap) == 0: # None left
				break
			d, sidx1, sidx2 = heapq.heappop(s_heap)
			choice = ( sidx1, sidx2 )
			print ('\tMerging closest state #' + str(choice[0]) + ' with state #' + str(choice[1]) + ' distance = ' + str(d))
			if choice in choices:
				continue
			choices.add(choice)
			midx -= 1

	print (datetime.datetime.now())
	# Now gather all merges and update triples in DTM
	merge_sets = {}
	for choice in choices:
		try:
			merge_sets[choice[0]].add(choice[1])
		except KeyError:
			merge_sets[choice[0]] = set()
			merge_sets[choice[0]].add(choice[1])
		try:
			merge_sets[choice[1]].add(choice[0])
		except KeyError:
			merge_sets[choice[1]] = set()
			merge_sets[choice[1]].add(choice[0])

	print ('Merging the states...')
	new_trips = list()
	for key_m, value in merge_sets.items():
		# key is an cs_idx
		# value is a set of cs_idxs
		for key_n in value:
			# Creating new triples for merger
			sidx1 = key_m
			sidx2 = key_n
			for trip_idx in graph[4][sidx2]:
				trip = dtm.triples[trip_idx]
				new_trip = ( sidx1, trip.action, trip.dest, trip.prob )
				new_trips.append(new_trip)

	# We sum up the probabilities of all occurances of new triples due to merging.
	print ('\tUpdating dtm...')
	update = {}
	for new_trip in new_trips:
		sidx, aidx, didx, prob = new_trip
		try:
			update[sidx].add(aidx)
		except KeyError:
			update[sidx] = set()
			update[sidx].add(aidx)
		trip_idx = graph[0][sidx][aidx][didx]
		if trip_idx == -1: # Create and add to dtm
			trip = pydtm.triple()
			trip.source = sidx
			trip.action = aidx
			trip.dest = didx
			trip.prob = 0
			trip_idx = len(dtm.triples)
			dtm.triples.append(trip)
			graph[0][sidx][aidx][didx] = trip_idx
			graph[1][sidx][aidx].append(trip_idx)
			graph[2][sidx][didx].append(trip_idx)
			graph[3][didx][sidx].append(trip_idx)
			graph[4][sidx].append(trip_idx)
		else:
			trip = dtm.triples[trip_idx]
		trip.prob += prob # will normalize 
	
	# Normalizing	
	for sidx, value in update.items():
		# value is set of aidx
		for aidx in value:
			total = 0.0
			for trip_idx in graph[1][sidx][aidx]:
				total += dtm.triples[trip_idx].prob
			for trip_idx in graph[1][sidx][aidx]:
				dtm.triples[trip_idx].prob = dtm.triples[trip_idx].prob / total

	print (datetime.datetime.now())
	return ( ( dtm, graph ) )


def test(dtmfile, trajfile, actionscsv):
	print (datetime.datetime.now())
	print ('Loading dtm...')
	dtm = pydtm.pydtm(dtmfile)
	print (datetime.datetime.now())
	print ('Creating dtm graphs...')
	graph = dtm_neighborhood.DTM_makeGraph(dtm)
	print (datetime.datetime.now())
	print ('Loading trajectories...')
	trajs = dtm_irl.DTM_loadTrajectoriesFile(dtm, trajfile, actionscsv)
	print (datetime.datetime.now())	
	print ('Creating new dtm with trajectory state merger...')
	answer = DTM_computeMergeStates(dtm, graph, trajs, True, 100)
	print (datetime.datetime.now())	
	return (answer)
