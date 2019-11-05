#testing opening the PolyASite2.0 datasets for human, mouse, c elegans
#AML 10/21/2019

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_colwidth', -1)

colnames = ["seqName",  "start" , "end",  "clusterID",  "avgTPM",  "strand",   "percentSupporting",   "protocolsSupporting",  "avgTPM2",   "type",   "upstreamClusters"]
pas_stuff =pd.read_csv('atlas.clusters.hg38.2-0.bed',delimiter='\t', names = colnames)
print ("0 based indexing") 
#print (type(pas_stuff))
#print (pas_stuff.info())

#first column
#chr_name = pas_stuff[pas_stuff.columns[0]]
#print (len(set(chr_name)))
'''
#note that the original BED file has 91 seqName values because 9 and '9' are both used in the original dataset.  There are only 90 chromosomes/contigs that have PAS clusters in the dataset
for x in set(chr_name):
	if type(x) != type("string"):
		print ("TRUE ", x)
	else:
		print (x)
chr_name_str = list(set([str(x) for x in chr_name]))

chr_name_str = sorted(chr_name_str)
print (chr_name_str)
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
'''

#print (len(chr_name.unique()))

#counting occurences on each named region
#print ( pas_stuff.columns ) 
# edf.type.value_counts()
'''
#saving the count of PAS clusters in each genomic region
storage = open("humanPolyASite2.0PASClustersPerGenomicRegion.txt", "w")
chr_dist = pas_stuff.seqName.value_counts()
storage.write("Region name   Number Clusters" + "\n")
storage.write(chr_dist.to_string())
storage.close()
'''

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
'''

