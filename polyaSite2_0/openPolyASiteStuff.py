#testing opening the PolyASite2.0 datasets for human, mouse, c elegans
#AML 10/21/2019

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pas_stuff =pd.read_csv('atlas.clusters.ce11.2-0.bed',delimiter='\t')
print(pas_stuff.info())
print ("0 based indexing") 

#print (" Chromosome_name  Start_pas_cluster  End_pas_cluster  Unique_cluster_ID  Avg_exp_TPM  Strand_of_cluster  Percentage_samples_supporting  Number_protocols_supp  Avg_exp_TPM  Cluster_type  upstream_clusters  ")

#Looking at unique chromsome/genbank IDs of regions PAS clusters were predicted in
print (pas_stuff.info())
chr_name = pas_stuff[pas_stuff.columns[0]]
print (chr_name.unique())
avg_TPM_all = pas_stuff[pas_stuff.columns[4]]
percent_supporting = pas_stuff[pas_stuff.columns[6]]
print ("Max TPM value: ", max(avg_TPM_all))
print ("Max percent usage value : ", max(percent_supporting))

'''
#graph the distribution of TPM values
avg_TPM_sorted = sorted(avg_TPM_all)

#dummy vals 
dummyX = np.arange(0,len(avg_TPM_sorted))
plt.scatter(dummyX,avg_TPM_sorted)
plt.xlabel("PAS clusters")
plt.ylabel("TPM values")
plt.title("C. Elegans PAS Cluster TPM Distribution")
plt.show()

sorted_percents = sorted(percent_supporting)
dummyX2 = np.arange(0,len(sorted_percents))
plt.scatter(dummyX2,sorted_percents)
plt.xlabel("PAS clusters")
plt.ylabel("Percentage of samples supporting the cluster")
plt.title("C. Elegans PAS Cluster Usage Distribution")
plt.show()


plt.scatter(avg_TPM_all, percent_supporting)
plt.xlabel("TPM for PAS Cluster")
plt.ylabel("Percentage of samples supporting the cluster")
plt.title("C. Elegans PAS Cluster TPM vs Percent samples supporting the cluster")
plt.show()

n, bins, patches = plt.hist(avg_TPM_all, 100)
plt.title("C Elegans TPM Distribution")
plt.show()

n, bins, patches = plt.hist(percent_supporting, 100)
plt.title("C Elegans Percent Samples Supporting a PAS Cluster Distribution")
plt.show()
'''
