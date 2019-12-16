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
import copy

from aparent.predictor import *
##################################################
#import bioPython for working with FASTA files
from Bio import SeqIO
##################################################

aparent_model = load_model('./saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
#plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
aparent_encoder = get_aparent_encoder(lib_bias=4)


#use to correct the indexes for the RC predictions so that they can be matched with the true cluster labels (since those are indexed according to the forward strand)
def flipSequenceIndex(index, lenSeq):
	b = lenSeq - 1
	return -1 * index + b
	
	
def placePeaksWithTolerance(peaks, clusters, tolerance, sign, lenSeq):
	clusterRanges = clusters.keys()
	for peak in peaks:
		if sign == "-":
			peak = flipSequenceIndex(peak, lenSeq)
		placed = False
		for rng in clusterRanges:
			if rng != 'Unplaced':
				lower = rng[0] - tolerance
				upper = rng[1] + tolerance
				if peak >= lower and peak <= upper: #if the peak is in [start, end]
					clusters[rng].append(peak)
					placed = True
					break
		if not placed: #wasn't placed
			clusters['Unplaced'].append(peak)
	return clusters

#using the peak-cluster dictionaries 
#need to count TP, FP, FN (placed peaks, unplaced peaks, empty clusters)
def fillConfMatrix(dictForward, dictRC):
	countTP = 0 #peaks in cluster
	countFP = 0 #peak outside of cluster (in ['Unplaced'])
	countFN = 0 #those w/no peak in the clusters 
	for key in dictForward:
		if key != 'Unplaced':
			inCluster = len(dictForward[key])
			if inCluster != 0:
				countTP += inCluster
				#print (key, " contains: ", inCluster)
			else:
				countFN += 1
		else: #unplaced peaks
			countFP += len(dictForward[key])
	for key in dictRC:
		if key != 'Unplaced':
			inCluster = len(dictRC[key])
			if inCluster != 0:
				countTP += inCluster
				#print (key, " contains: ", inCluster)
			else:
				countFN += 1
		else: #unplaced peaks
			countFP += len(dictRC[key])
	return countTP, countFP, countFN
	
def calculatePrecisionRecall(tp, fp, fn):
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	if tp == 0:
		return precision, recall, None
	else: 
		f1 =  (2 * (precision * recall))/(precision + recall)
	return precision, recall, f1

def openForwardReverse(stem, name):
	totalNameFor = stem + name + ".npy"
	print (totalNameFor)
	forward = np.load(totalNameFor)
	reverse = np.load(stem + name + "RC.npy")
	return forward, reverse


def openTrueValuesForType(name, pasType):
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
	if pasType == "All":
		for index, row in currentTrueVals.iterrows():
			if row['strand'] == "+": #forward strand
				clustersForward[(row['start'], row['end'])] = []
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
	return clustersForward, clustersRC



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

types = ['All', 'IN', 'TE', 'IG', 'AI', 'EX', 'DS', 'AE', 'AU'] #for each type and subtype
#types = ['AU', 'AE']
tolerances = [0, 10, 20] #tolerances around clusters
dists = [50] #for peak finding algorithm
minHeights = np.linspace(0,1.0,20) #skip 0 because it's going to be bad runtime wise
min_dist = 50
peak_prom = (0.01, None)

iterations = {}
largestF1 = float('-inf')
bestSettings = ""
fileNames = ["chr15"]
names = ["15"]
minHeights = [0.01]
tolerances = [0, 10, 20] #tolerances around clusters
dists = [50] #for peak finding algorithm #skip 0 because it's going to be bad runtime wise
peak_prom = (0.01, None)


#dictionary set up
for pasType in types:
	iterations[pasType] = {}
	for tolerance in tolerances:
		for dist in dists:
			iterations[pasType][(tolerance,dist)] = []


stem = "./chromosomePredictions50/"
name = "Y"
nameStem = "chrY"
min_height = 0.01


pasTypeTotals = {}
for pasType in types:
	counterTypes = 0
	for i, fname in enumerate(fileNames):
		#add overall types to iterations:
		forward, reverse = openForwardReverse(stem, fname)
		f = open(names[i] + pasType + "ConfusionMatrices.txt", "w")
		print ("-------------------------------------------------------------")
		print ("Chromosome: ", names[i], " PAS Type: ", pasType)
		print ("length: ", forward.shape)
		clustersForward, clustersRC = openTrueValuesForType(names[i], pasType)
		print ("Forward size: ", len(clustersForward.keys()) - 1)
		print ("Reverse size: ", len(clustersRC.keys()) -1 )
		print ("All signals: ", len(clustersForward.keys()) + len(clustersRC.keys())- 2)
		counterTypes += len(clustersForward.keys()) + len(clustersRC.keys())- 2
		for tolerance in tolerances:
			for dist in dists:
				dummyList = []
				for minh in minHeights:
					if minh != 0:
						clustersForwardcopy = copy.deepcopy(clustersForward)
						clustersRCcopy = copy.deepcopy(clustersRC)
						forwardPeaks = find_peaks_ChromosomeVersion(forward, minh, dist, (0.01, None)) 
						reversePeaks = find_peaks_ChromosomeVersion(reverse, minh, dist, (0.01, None)) 
						print ("Number forward peaks: ", forwardPeaks.shape)
						print ("Number forward peaks: ", reversePeaks.shape)
						print ("Total peaks: ",  forwardPeaks.shape[0] + reversePeaks.shape[0])
						clustersForTol = placePeaksWithTolerance(forwardPeaks, clustersForwardcopy, tolerance,  "+", forward.shape[0])	
						clustersRCTol = placePeaksWithTolerance(reversePeaks, clustersRCcopy, tolerance,  "-", forward.shape[0])	
						countTP, countFP, countFN = fillConfMatrix(clustersForTol, clustersRCTol)
						print (countTP, countFP, countFN)
						dummyList.append((countTP,countFP,countFN))
						#print ("For min peak height: ", minh, " TP: ", countTP, " FP: ", countFP, " FN: ", countFN)
						print ("tolerance: ", tolerance, " dist: ", dist, " minh: ", minh)
						fileLine = "tolerance: " + str(tolerance) + " dist: " + str( dist) +  " minh: " + str(minh) + " TP: " + str(countTP) + " FP: " + str(countFP) + " FN: " + str(countFN) + "\n"
						f.write(fileLine)
						#print ("-------------------------------------------------------------")
				iterations[pasType][(tolerance,dist)].append(dummyList) #append list of TP, FP, FN for each chromosome examined
				#print (dummyList)
				#print (iterations)
		f.close()
	pasTypeTotals[pasType] = counterTypes

for k in pasTypeTotals.keys():
	print (k, " " , pasTypeTotals[k])
	
	
	
			
		




		
	
	
 






