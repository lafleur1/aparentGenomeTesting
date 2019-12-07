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
#loading model

start_time = time.time()
x,y = find_polya_peaks_memoryFriendly(
                aparent_model,
                aparent_encoder,
                seq,
                sequence_stride=15,
                conv_smoothing=False,
                peak_min_height=0.01,
                peak_min_distance=50,
                peak_prominence=(0.01, None)
            )
secondsDiffs = time.time()-start_time
print (secondsDiffs)

#############
# 2
print ("METHOD 2: ")
#loading model
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
##################3
#3 
#find_polya_peaks_memoryFriendlyV3(aparent_model, aparent_encoder, seq, sequence_stride=10, output_size = 10000, fileStem) :
fileStem = "dummy/dummy2"
files = find_polya_peaks_memoryFriendlyV3(aparent_model, aparent_encoder, seq, fileStem, 15, 100000)
print ("original last 100: ")
print (list(y[0][-100:]))
#assemble files and check that output is the same
total_preds = np.zeros(0)
for f in files:
	opened_array = np.load(f + ".npy")
	print (opened_array.shape)
	total_preds = np.concatenate((total_preds,  opened_array), axis = 0)               
print ("Assembled shape: ", total_preds.shape)
print ("Original shape: ", y[0].shape)

print ("Original vs V2: ")
print (np.array_equal(y2, y[0]))
print ("Original vs assmbled: ")
print (np.array_equal(y[0], total_preds))


"""
#plot all to see difference
x_data = np.arange(0,len(seq))
plt.plot(x_data, y[0])
plt.show()
plt.plot(x_data, y2)
plt.show()
plt.plot(x_data, total_preds)
plt.show()
"""


#finding where the 
for i in range(0,total_preds.shape[0]):
	#print ("---------------------------------")
	#print ("orig: ", y[0][i])
	#print ("v2: ", y2[i])
	#print ("v3: ", total_preds[i])
	#print ("---------------------------------")
	if y[0][i] == y2[i] and y2[i] == total_preds[i]:
		dummy = 0
	else:
		print ("MISMATCH ", i)
		break

	


