#testing opening the PolyASite2.0 datasets for human, mouse, c elegans
#AML 10/21/2019

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections  as mc

pd.set_option('display.max_colwidth', -1)

colnames = ["seqName",  "start" , "end",  "clusterID",  "avgTPM",  "strand",   "percentSupporting",   "protocolsSupporting",  "avgTPM2",   "type",   "upstreamClusters"]
pas_stuff =pd.read_csv('atlas.clusters.hg38.2-0.bed',delimiter='\t', names = colnames) #human
#pas_stuff =pd.read_csv('atlas.clusters.ce11.2-0.bed',delimiter='\t', names = colnames) # c elegans
print ("0 based indexing for positions!!!!") 

onGL000219 = pas_stuff['seqName'] == "GL000219.1"
onGLSites = pas_stuff[onGL000219]
startSites = onGLSites['start']
#print (startSites)
print (min(startSites))
endSites = onGLSites['end']
print (max(endSites))
onGLSites.sort_values("start")
print (onGLSites.head())

siteSizes = []
for index,row in pas_stuff.iterrows():
	siteSizes.append(row['end'] - row['start'])

#print (linesUsage)
sizeDict = {}
#lengths = sorted(list(set(siteSizes)))
for site in siteSizes:
	if site in sizeDict.keys():
		sizeDict[site] += 1
	else:
		sizeDict[site] = 1
print (sizeDict)

print (sorted(sizeDict.keys()))


sizesPossCounts = [sizeDict[x] for x in sizesPossible]

plt.scatter(sizesPossible, sizesPossCounts)
plt.show()



onGL000219 = pas_stuff['seqName'] == "GL000219.1"

onGLSites = pas_stuff[onGL000219]
print (onGLSites)

linesUsage = []
linesTPM = []
siteSizes = []
for index,row in onGLSites.iterrows():
	linesTPM.append([(row['start'], row['avgTPM']), (row['end'], row['avgTPM'])])
	linesUsage.append([(row['start'], row['percentSupporting']), (row['end'], row['percentSupporting'])])
	siteSizes.append(row['end'] - row['start'])


#print (onGLSites)

startSites = onGLSites['start']
#print (startSites)
print (min(startSites))
endSites = onGLSites['end']
print (max(endSites))

print (onGL000219 = pas_stuff['seqName'] == "GL000219.1")

#manually create data for the region to graph
#graphing TPM  and usage percentages as line segment graphs w/ matplotlib
#used this post: https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors 

"""
lcUsage = mc.LineCollection(linesUsage, linewidths=2)
lcTPM = mc.LineCollection(linesTPM, linewidths=2)

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.add_collection(lcUsage)
ax1.autoscale()
ax1.set_title("Percent usage")

ax2 = fig.add_subplot()
ax2.add_collection(lcTPM)
ax2.autoscale()
ax2.set_title("TPM")
plt.show()

#fig, axs = plt.subplots(2)
"""


