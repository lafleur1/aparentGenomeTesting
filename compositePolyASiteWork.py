#composite polyASite2.0 investigations
#AML 11/19


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections  as mc

#open the total human dataset 
colnames = ["seqName",  "start" , "end",  "clusterID",  "avgTPM",  "strand",   "percentSupporting",   "protocolsSupporting",  "avgTPM2",   "type",   "upstreamClusters"]
pas_stuff =pd.read_csv('atlas.clusters.hg38.2-0.bed',delimiter='\t', names = colnames) 

#working with sites from just one unlocalized region 
onGL000219 = pas_stuff['seqName'] == "GL000219.1"
onGLSites = pas_stuff[onGL000219]
startSites = onGLSites['start']
endSites = onGLSites['end']
onGLSites.sort_values("start")

############## Counting numbe of all 'true' sites in the entire genome ########################
"""
siteSizes = []
totalPos = 0
for index,row in pas_stuff.iterrows():
	siteSizes.append(row['end'] - row['start'])
	totalPos += row['end'] - row['start']

print ("Total lenght of all sites in the database: ", totalPos)
#total number of all "true positives" in the genome is 5425800
"""
#############################3
################ Plotting the distribution of site sizes in polyASite2.0 ##########################
"""
sizeDict = {}
totalSize = 0
for site in siteSizes:
	if site in sizeDict.keys():
		sizeDict[site] += 1
	else:
		sizeDict[site] = 1

sizesPossible = sorted(sizeDict.keys())
sizesPossCounts = [sizeDict[x] for x in sizesPossible]
plt.plot(sizesPossible, sizesPossCounts)
plt.title("PolyASite2.0 APA Site Sizes")
plt.xlabel("Length of site")
plt.ylabel("Number of sites")
plt.show()
"""
##################################
########### Plotting the usage and TPM values for just one region ###############
"""
linesUsage = []
linesTPM = []
siteSizes = []
for index,row in onGLSites.iterrows():
	linesTPM.append([(row['start'], row['avgTPM']), (row['end'], row['avgTPM'])])
	linesUsage.append([(row['start'], row['percentSupporting']), (row['end'], row['percentSupporting'])])
	siteSizes.append(row['end'] - row['start'])

#manually create data for the region to graph
#graphing TPM  and usage percentages as line segment graphs w/ matplotlib
#used this post: https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors 

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
"""
########################################## counting region types in genome ###########################
#first column
"""
chr_name = pas_stuff[pas_stuff.columns[0]]
#note that the original BED file has 91 seqName values because 9 and '9' are both used in the original dataset.  There are only 90 chromosomes/contigs that have PAS clusters in the dataset
allStuff = list(set([str(x) for x in chr_name ]))
chr_name_str = list(set([str(x) for x in chr_name if "GL" in str(x) or "KI" in str(x)]))
others = list(set([str(x) for x in chr_name if "GL" not in str(x) and "KI" not in str(x)]))
sorted(chr_name_str)
print (chr_name_str)
print (len(chr_name_str))
print (len(others))
print (len(allStuff))
print (others)
file_out = open("UnlocalizedNames.txt", "w")
for name in chr_name_str:
	file_out.write(name + "\n")
file_out.close()
"""
"""
chr_name_str = sorted(chr_name_str)
#figure out types of unlocalized in a chromosome vs unplaced contigs:
numberGL = 0
numberKI = 0
other = 0

for name in chr_name_str:
	if "KI" in name:
		numberKI += 1
	elif "GL" in name:
		numberGL += 1
	else:
		other += 1
print ("Number chromosome unlocalized genomic contigs: ", numberGL)
print ("Number unplaced genomic contigs: ", numberKI)
print ("All other regions: ", other)
print ("SUM: ", other + numberKI + numberGL)
print ("ACTUAL: ", len(chr_name_str))
"""
################# storing names of region and number PAS sites ###########3
"""
#saving the count of PAS clusters in each genomic region
#storage = open("humanPolyASite2.0PASClustersPerGenomicRegion.txt", "w")
chr_dist = pas_stuff.seqName.value_counts()
#storage.write("Region name   Number Clusters" + "\n")
#storage.write(chr_dist.to_string())
#storage.close()
chr_dist.plot.bar()
plt.title("Clusters per Genomic Region")
plt.ylabel("Cluster Count")
plt.xlabel("Genomic Region")
plt.show()
"""
'''
################# graphing type of clusters in human genome ###################3
type_dist = pas_stuff.type.value_counts()
#ax = type_dist.plot.bar(x='type', rot = 0)
print (type(type_dist))
type_dist.plot.bar()
plt.title("Human polyA Site Clusters by Type")
plt.ylabel("Cluster Count")
plt.xlabel("Type")
plt.show()
'''
###################################################Exploring PAS Cluster distributions ###########################################
avg_TPM_all = pas_stuff[pas_stuff.columns[4]]
percent_supporting = pas_stuff[pas_stuff.columns[6]]
print ("Max TPM value: ", max(avg_TPM_all))
print ("Max percent usage value : ", max(percent_supporting))

print (pas_stuff.info())
#graph the distribution of TPM values
avg_TPM_sorted = sorted(avg_TPM_all)
'''
'''

count_percents = pas_stuff.percentSupporting.value_counts()
count_percents.plot.bar()
plt.show()
"""
print (count_percents)

print (len(avg_TPM_sorted))
#dummy vals 
dummyX = np.arange(0,len(avg_TPM_sorted))
plt.scatter(dummyX,avg_TPM_sorted)
plt.xlabel("PAS clusters")
plt.ylabel("TPM values")
plt.title("Human PAS Cluster TPM Distribution")
plt.show()

sorted_percents = sorted(percent_supporting)
dummyX2 = np.arange(0,len(sorted_percents))
plt.scatter(dummyX2,sorted_percents)
plt.xlabel("PAS clusters")
plt.ylabel("Percentage of samples supporting the cluster")
plt.title("Human PAS Cluster Usage Distribution")
plt.show()


plt.scatter(avg_TPM_all, percent_supporting)
plt.xlabel("TPM for PAS Cluster")
plt.ylabel("Percentage of samples supporting the cluster")
plt.title("Human PAS Cluster TPM vs Percent samples supporting the cluster")
plt.show()

n, bins, patches = plt.hist(avg_TPM_all, 100)
plt.title("Human TPM Distribution")
plt.show()

n, bins, patches = plt.hist(percent_supporting, 20)
plt.title("Human Percent Samples Supporting a PAS Cluster Distribution")
plt.show()
"""



#check size vs APA distributions
#look at signals throughout a gene
#do peaks in/out of clusters
#ranking strenghts -> 
#see intervals vs curves
#play around with clsuter
#Try just UTR stuff
#run reverse complements for negative/positives


##################################################
