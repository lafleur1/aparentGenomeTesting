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


#use to correct the indexes for the RC predictions so that they can be matched with the true cluster labels (since those are indexed according to the forward strand)
def flipSequenceIndex(index, lenSeq):
	b = lenSeq - 1
	return -1 * index + b
	

def openForwardReverse(stem, name):
	totalNameFor = stem + name + ".npy"
	forward = np.load(totalNameFor)
	reverse = np.load(stem + name + "RC.npy")
	return forward, reverse
	
	

stem = "./chromosomePredictions50/"
name = "Y"
fileName = "chrY"
min_height = 0.01
heights = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08, 0.09, 0.1]
tolerances = [0, 20]
dists = [25,50,75]
min_dist = 50
peak_prom = (0.01, None)


#open forward and reverse predictions
forward, reverse = openForwardReverse(stem, fileName)
types = ['All', 'IN', 'TE', 'IG', 'AI', 'EX', 'DS', 'AE', 'AU'] #for each type and subtype
