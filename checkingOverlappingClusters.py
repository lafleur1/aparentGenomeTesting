#checking to see if any clusters overlap in the atlas file

import pandas as pd
import queue as q

#opening all the true values from PolyASite2.0 
colnames = ["seqName",  "start" , "end",  "clusterID",  "avgTPM",  "strand",   "percentSupporting",   "protocolsSupporting",  "avgTPM2",   "type",   "upstreamClusters"]
pas_stuff =pd.read_csv('atlas.clusters.hg38.2-0.bed',delimiter='\t', names = colnames) 



names = list(set([x for x in pas_stuff['seqName']]))
for name in names:
	print ("ON: ", name)
	list_clusters = []
	pq_clusters = q.PriorityQueue()
	types =q.PriorityQueue()
	trueValBoolMask = pas_stuff['seqName'] == name
	currentTrueVals = pas_stuff[trueValBoolMask] #filtered true vals
	currentTrueValsMask = currentTrueVals['strand'] == "+"
	print ("ON + STRAND")
	forwardVals = currentTrueVals[currentTrueValsMask]
	for index, row in forwardVals.iterrows():
		#list_clusters.append((row['start'], row['end']))
		types.put(row['type'], row['start'])
		pq_clusters.put((row['start'], row['end']), row['start'])
	print ("Filled queues")
	if not pq_clusters.empty():
		currentCluster = pq_clusters.get() #start with first cluster
		currentType = types.get()
		while pq_clusters.empty():
			otherCluster = pq_clusters.get()
			otherType = types.get()
			if otherCluster[0] < currentCluster[1]:
				print ("Overlap has occured! On region: ", name, " with ranges: ", currentCluster, " " , currentType, " ", otherCluster[0], " ", otherType)
				
				break
			else:
				currentCluster = otherCluster
				currentType = otherType
	pq_clusters2 = q.PriorityQueue()
	types2 =q.PriorityQueue()
	trueValBoolMask = pas_stuff['seqName'] == name
	currentTrueVals = pas_stuff[trueValBoolMask] #filtered true vals
	currentTrueValsMask = currentTrueVals['strand'] == "-"
	print ("ON - STRAND")
	forwardVals = currentTrueVals[currentTrueValsMask]
	for index, row in forwardVals.iterrows():
		#list_clusters.append((row['start'], row['end']))
		types2.put(row['type'], row['start'])
		pq_clusters2.put((row['start'], row['end']), row['start'])
	print ("Filled queues")
	if not pq_clusters2.empty():
		currentCluster = pq_clusters2.get() #start with first cluster
		currentType = types2.get()
		while pq_clusters2.empty():
			otherCluster = pq_clusters2.get()
			otherType = types2.get()
			if otherCluster[0] < currentCluster[1]:
				print ("Overlap has occured! On region: ", name, " with ranges: ", currentCluster, " " , currentType, " ", otherCluster[0], " ", otherType)
				
				break
			else:
				currentCluster = otherCluster
				currentType = otherType
