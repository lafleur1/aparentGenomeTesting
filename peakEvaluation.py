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

###############3
#method 1
#print ("METHOD 1: ")
fastaDestination = "./fastas/"
name = "GL000225.1"
contigSeq = SeqIO.read(fastaDestination + name + ".fasta", "fasta")
seq = contigSeq.seq #actual genomic sequence from the file
print (len(seq))


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

"""	
#loading model
#scoring forward
start_time = time.time()
x,y2 = find_polya_peaks_memoryFriendlyV2_LOUD(
                aparent_model,
                aparent_encoder,
                seq,
                sequence_stride=15,
                conv_smoothing=False,
                peak_min_height=0.01,
                peak_min_distance=50,
                peak_prominence=(0.01, None),
                counter = 100000
            )
secondsDiffs = time.time()-start_time
print (secondsDiffs)
outsidePeak = find_peaks_ChromosomeVersion(y2, 0.01, 50, (0.01, None)) #tested 


#log odds of the peaks
peak_iso_scores = score_polya_peaks(
    aparent_model,
    aparent_encoder,
    seq,
    outsidePeak,
    sequence_stride=1,
    strided_agg_mode='max',
    iso_scoring_mode='both',
    score_unit='log'
)

print ("scores: ")
print (peak_iso_scores) 


#get reverse complement 
rcseq = contigSeq.reverse_complement()
start_time = time.time()
x,y2RC = find_polya_peaks_memoryFriendlyV2_LOUD(
                aparent_model,
                aparent_encoder,
                rcseq,
                sequence_stride=15,
                conv_smoothing=False,
                peak_min_height=0.01,
                peak_min_distance=50,
                peak_prominence=(0.01, None),
                counter = 100000
            )
secondsDiffs = time.time()-start_time
print (secondsDiffs)
outsidePeakRC = find_peaks_ChromosomeVersion(y2RC, 0.01, 50, (0.01, None)) #tested 
"""

def openForwardReverse(stem, name):
	forward = np.load(stem + name + ".npy"
	reverse = np.load(stem + name + "RC.npy")
	return forward, reverse

forward = np.load("./chromosomePredictions50/chrY.npy")
reverse = np.load("./chromosomePredictions50/chrYRC.npy")
print ("forward: ", forward.shape)
print ("reverse: ", reverse.shape)
outsidePeak = find_peaks_ChromosomeVersion(forward, 0.01, 50, (0.01, None)) #tested
outsidePeakRC = find_peaks_ChromosomeVersion(reverse, 0.01, 50, (0.01, None)) #tested 
print ("outside peak shape: ", outsidePeak.shape)
print ("RC shape: ", outsidePeakRC.shape)

#opening all the true values from PolyASite2.0 
colnames = ["seqName",  "start" , "end",  "clusterID",  "avgTPM",  "strand",   "percentSupporting",   "protocolsSupporting",  "avgTPM2",   "type",   "upstreamClusters"]
pas_stuff =pd.read_csv('atlas.clusters.hg38.2-0.bed',delimiter='\t', names = colnames) 
trueValBoolMask = pas_stuff['seqName'] == "Y"
currentTrueVals = pas_stuff[trueValBoolMask] #filtered true vals


#set up true value array 
clustersForward = {}
clustersRC = {} #key of (Start, End) will use to track clusters with no peaks for the FN  
#OR: 
print (currentTrueVals)
for index, row in currentTrueVals.iterrows():
	if row['strand'] == "+": #forward strand
		clustersForward[(row['start'], row['end'])] = []
	else: #negative strand, RC cluster
		clustersRC[(row['start'], row['end'])] = []


clustersForward['Unplaced'] = []
clustersRC['Unplaced'] = []
#print (clustersForward)
#print (clustersRC)
#fill clusters Forward
#print ("peak shape: ", outsidePeak.shape)
cFRanges = clustersForward.keys()

"""
for peak in outsidePeak:
	placed = False
	for rng in cFRanges:
		if rng != 'Unplaced' and peak >= rng[0] and peak <= rng[1]: #if the peak is in [start, end]
			clustersForward[rng].append(peak)
			placed = True
			break
	if not placed: #wasn't placed
		clustersForward['Unplaced'].append(peak)
"""


	
	
clustersFor0Tol = placePeaksWithTolerance(outsidePeak, clustersForward, 0,  "+", forward.shape[0])	
clustersRC0Tol = placePeaksWithTolerance(outsidePeakRC, clustersRC, 0,  "-", forward.shape[0])	
print (fillConfMatrix(clustersFor0Tol, clustersRC0Tol))
#calculate true values for the threshold used for the peaks
cutoffs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


"""
cRCRanges = clustersRC.keys()
for peak in outsidePeakRC:
	placed = False
	for rng in cRCRanges:
		if rng != 'Unplaced' and peak >= rng[0] and peak <= rng[1]: #if the peak is in [start, end]
			clustersRC[rng].append(peak)
			placed = True
			break
	if not placed: #wasn't placed
		clustersRC['Unplaced'].append(peak)
	"""	



		
	
	
 






