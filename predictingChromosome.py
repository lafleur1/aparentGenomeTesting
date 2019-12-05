#predicting chromosome + reverse complement
#AML
#using memoryFriendly V3 prediction method (stores chromosome in slices, simply add all slices together to recover full chromosome for findng peaks

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

#open chro
#doing y, x and their RC's

fastaDestination = "./fastas/"
fastaNames = ["chrY" , "chr22", "chr21"]
#fastaNames = ["GL000225.1"]
stem = "chromosomePredictions50/"

for name in fastaNames:
	contigSeq = SeqIO.read(fastaDestination + name + ".fasta", "fasta")
	seq = contigSeq.seq #actual genomic sequence from the file
	rcseq = contigSeq.reverse_complement()
	print (name, " ", len(seq), " ", len(rcseq))
	fileStem = stem + name
	fileStemRC = stem + name + "RC"
	print ("FORWARD")
	x,y = find_polya_peaks_memoryFriendlyV2_LOUD(
                aparent_model,
                aparent_encoder,
                seq,
                sequence_stride=15,
                conv_smoothing=False,
                peak_min_height=0.01,
                peak_min_distance=50,
                peak_prominence=(0.01, None),
                counter = 500000
            )
	print ("REVERSE COMPLEMENT")
	xRC,yRC = find_polya_peaks_memoryFriendlyV2_LOUD(
                aparent_model,
                aparent_encoder,
                rcseq,
                sequence_stride=15,
                conv_smoothing=False,
                peak_min_height=0.01,
                peak_min_distance=50,
                peak_prominence=(0.01, None),
                counter = 500000
            )
	np.save(fileStem, y)
	np.save(fileStemRC, yRC)
	print ("FINISHED WITH : ", name)
	


