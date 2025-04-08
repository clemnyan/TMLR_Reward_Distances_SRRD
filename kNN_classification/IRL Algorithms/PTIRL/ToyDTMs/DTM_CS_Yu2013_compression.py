#!/usr/bin/python
# Compress a csv and generate a new one.
# argv[1] is the filename of the csv to be clustered with kmeans
# argv[2] is the number of clusters
# argv[3] is the filename of the csv to write the new clusters.
import sys
filename = sys.argv[1]
clusters = int(sys.argv[2])
savefilename = sys.argv[3]
import csv
import numpy as np
from sklearn.cluster import KMeans

with open(filename, 'r') as csvfile:
	reader = csv.reader(csvfile)
	csvlist = list(reader);

	_header = csvlist.pop(0)
	new_list = list()
	for _list in csvlist:
		_temp = list()
		for i in _list:
			try:
				float(i)
				_temp.append(float(i))
			except ValueError:
				_temp.append(0.0)
		new_list.append(_temp);

#	print new_list
	X = np.array(new_list,dtype=float)
#	print X[1]
#	print X[2]
	print "Starting KMeans.."
	kmeans = KMeans(n_clusters=clusters).fit(X)
	final_list = kmeans.cluster_centers_.tolist()
	_final_list = list()
	_final_list.append(_header)
	final_list[:0] = _final_list
	with open(savefilename, 'w') as savefile:
		writer = csv.writer(savefile)
		writer.writerows(final_list)
