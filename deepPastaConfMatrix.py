#DeepPasta data examination with same methods for same regions and truth values 


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



def find_peaks_deepPASTA(chrname, allChrs):
	#for each chromosome, sep. + and -v
	forwardPeaks = []
	reversePeaks = []
	trueValBoolMask = allChrs['seqName'] == chrname
	#print (name)
	currentTrueVals = allChrs[trueValBoolMask] #filtered true vals
	for index, row in currentTrueVals.iterrows():
		if row['strand'] == "+":
			forwardPeaks.append(row['position'])
		elif row['strand'] == "-":
			reversePeaks.append(row['position'])
		else:
			print ("ERROR!  No strand associated!")
	return forwardPeaks, reversePeaks
		




#treat these as "peaks"

data = pd.read_csv("genome_wide_polyA_site_prediction_human_DeepPASTA.txt", sep="\t", header=None)
data.columns =  ["seqName", "position", "strand", "score"]

chromosomes = [ 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
namesPolyA = ["15", "16", "17", "18", "19", "20", "21", "22", "X", "Y"]
tolerances = [0, 10, 20] #tolerances around clusters


types = ['All', 'IN', 'TE', 'IG', 'AI', 'EX', 'DS', 'AE', 'AU'] #for each type and subtype



pasTypeTotals = {}
f = open( "deepPASTAConfusionMatrices.txt", "w")
for pasType in types:
	counterTypes = 0
	for tolerance in tolerances:
		countTPtotal = 0
		countFPtotal = 0
		countFNtotal = 0
		for i, fname in enumerate(chromosomes):
				print ("-------------------------------------------------------------")
				print ("Chromosome: ", namesPolyA[i], " PAS Type: ", pasType)
				clustersForward, clustersRC = openTrueValuesForType(namesPolyA[i], pasType)
				print ("Forward size: ", len(clustersForward.keys()) - 1)
				print ("Reverse size: ", len(clustersRC.keys()) -1 )
				print ("All signals: ", len(clustersForward.keys()) + len(clustersRC.keys())- 2)
				if tolerance == tolerances[0]:
					counterTypes += len(clustersForward.keys()) + len(clustersRC.keys())- 2
				forwardPeaks, reversePeaks = find_peaks_deepPASTA(fname, data)	 
				print ("Number forward peaks: ", len(forwardPeaks))
				print ("Number reverse peaks: ", len(reversePeaks))
				print ("Total peaks: ",   len(forwardPeaks)+ len(reversePeaks))
				clustersForTol = placePeaksWithTolerance(forwardPeaks, clustersForward, tolerance,  "+", len(forwardPeaks))	
				clustersRCTol = placePeaksWithTolerance(reversePeaks, clustersRC, tolerance,  "-", len(forwardPeaks))	
				countTP, countFP, countFN = fillConfMatrix(clustersForTol, clustersRCTol)
				print ("tolerance, fname: ", tolerance, " ", fname)
				print (countTP, countFP, countFN)
				countTPtotal += countTP
				countFPtotal += countFP
				countFNtotal += countFN 
				#print ("For min peak height: ", minh, " TP: ", countTP, " FP: ", countFP, " FN: ", countFN)
				print ("tolerance: ", tolerance)
		print ("Total TP: ", countTPtotal, " total FP: ", countFPtotal, " total FN: ", countFNtotal)
		fileLine = "PAS: " + pasType + " tolerance: " + str(tolerance) + " TP: " + str(countTPtotal) + " FP: " + str(countFPtotal) + " FN: " + str(countFNtotal) + "\n"
		f.write(fileLine)
	pasTypeTotals[pasType] = counterTypes
f.close()

for k in pasTypeTotals.keys():
	print (k, " " , pasTypeTotals[k])
	
	
	
			
		




		
	
	
 






