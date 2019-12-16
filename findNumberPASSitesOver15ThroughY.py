from __future__ import print_function
import keras
from keras.models import Sequential, Model, load_model
from keras import backend as K
import tensorflow as tf
import isolearn.keras as iso
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.utils import plot_model
import time

from aparent.predictor import *
##################################################
#import bioPython for working with FASTA files
from Bio import SeqIO
##################################################

aparent_model = load_model('./saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
#plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
aparent_encoder = get_aparent_encoder(lib_bias=4)

#def openForwardReverse(stem, name):
#	totalNameFor = stem + name + ".npy"
#	print (totalNameFor)
#	forward = np.load(totalNameFor)
#	reverse = np.load(stem + name + "RC.npy")
#	return forward, reverse


def openTrueValuesForTypeAndCount(name, pasType):
	#opening all the true values from PolyASite2.0 
	colnames = ["seqName",  "start" , "end",  "clusterID",  "avgTPM",  "strand",   "percentSupporting",   "protocolsSupporting",  "avgTPM2",   "type",   "upstreamClusters"]
	pas_stuff =pd.read_csv('atlas.clusters.hg38.2-0.bed',delimiter='\t', names = colnames, dtype = {"seqName": str}) 
	trueValBoolMask = pas_stuff['seqName'] == name
	#print (name)
	currentTrueVals = pas_stuff[trueValBoolMask] #filtered true vals
	#print (currentTrueVals)
	#set up true value array 
	clustersForward = {}
	clustersRC = {} #key of (Start, End) will use to track clusters with no peaks for the FN  
	totalLength = 0 
	if pasType == "All":
		for index, row in currentTrueVals.iterrows():
			if row['strand'] == "+": #forward strand
				clustersForward[(row['start'], row['end'])] = []
				totalLength += row['end']-row['start'] + 1
			else: #negative strand, RC cluster
				clustersRC[(row['start'], row['end'])] = []
		clustersForward['Unplaced'] = []
		clustersRC['Unplaced'] = []
	else:
		
		maskType = currentTrueVals["type"] == pasType
		maskedTrue = currentTrueVals[maskType]
		for index, row in maskedTrue.iterrows():
			if row['strand'] == "+": #forward strand
				clustersForward[(row['start'], row['end'])] = []
			else: #negative strand, RC cluster
				clustersRC[(row['start'], row['end'])] = []
		clustersForward['Unplaced'] = []
		clustersRC['Unplaced'] = []
	#print (clustersForward)	
	return clustersForward, clustersRC, totalLength, len(clustersForward.keys())



#https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2631518/


# https://ggbaker.ca/data-science/content/filtering.html
#https://ggbaker.ca/data-science/content/filtering.html 

#choose several filters from scipy.signal filters
#find peaks off of these filtered signals

#using this resource: testing smoothed vs not-smoothed for a few differnet smoothing methods, then signal prominence thresholding with different windows (size to use is waht????)
#vary none, smooth method 1, smooth method 2, smooth method 3, .....
#Vary prominence adnd min height cut offs....
#

#open forward and reverse predictions

types = ['All'] #for each type and subtype
#types = ['AU', 'AE']
tolerances = [0, 10, 20] #tolerances around clusters
dists = [50] #for peak finding algorithm
minHeights = np.linspace(0,1.0,20) #skip 0 because it's going to be bad runtime wise
min_dist = 50
peak_prom = (0.01, None)

iterations = {}
largestF1 = float('-inf')
bestSettings = ""
fileNames = ["chr15", "chr18", "chr19", "chr21", "chr22", "chrX", "chrY"]
names = ["15", "18", "19", "21", "22", "X", "Y"]
minHeights = [0.01, 0.05, 0.1]
tolerances = [0, 10, 20] #tolerances around clusters
dists = [25, 50] #for peak finding algorithm #skip 0 because it's going to be bad runtime wise
peak_prom = (0.01, None)



#dictionary set up
for pasType in types:
	iterations[pasType] = {}
	for tolerance in tolerances:
		for dist in dists:
			iterations[pasType][(tolerance,dist)] = []


stem = "./chromosomePredictions50/"
fastaDestination = "./fastas/"
name = "Y"
nameStem = "chrY"
min_height = 0.01

totalPAS = 0
totalPASLength = 0
totalSeqLen = 0
pasTypeTotals = {}
for pasType in types:
	counterTypes = 0
	for i, fname in enumerate(fileNames):
		#add overall types to iterations:
		#forward, reverse = openForwardReverse(stem, fname)
		contigSeq = SeqIO.read(fastaDestination + fname + ".fasta", "fasta")
		seq = contigSeq.seq #actual genomic sequence from the file
		rcseq = contigSeq.reverse_complement()
		print ("-------------------------------------------------------------")
		print ("Chromosome: ", names[i], " PAS Type: ", pasType)
		print ("length: ", len(seq))
		totalSeqLen += 2 * len(seq)
		clustersForward, clustersRC, countLen, countSites = openTrueValuesForTypeAndCount(names[i], pasType)
		print ("Forward size: ", len(clustersForward.keys()) - 1)
		print ("Reverse size: ", len(clustersRC.keys()) -1 )
		print ("All signals: ", len(clustersForward.keys()) + len(clustersRC.keys())- 2)
		totalPAS += len(clustersForward.keys()) + len(clustersRC.keys())
		totalPASLength += 	countLen


print ("total PAS sites: ", totalPAS)
print ("total length of all sites: ", totalPASLength)
print ("Length of sequence: ", totalSeqLen)
print ("PAS fraction tol 0: ", 	totalPASLength/totalSeqLen)
print ("with tolerance of 10: ", (totalPAS * 20 + totalPASLength)/totalSeqLen)
print ("With tolerance of 20: ", (totalPAS * 40 + totalPASLength)/totalSeqLen)

	
	

		
			
		




		
	
	
 






