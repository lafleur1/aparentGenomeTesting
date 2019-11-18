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


from aparent.predictor import *
##################################################
#import bioPython for working with FASTA files
from Bio import SeqIO
##################################################


#loading model
aparent_model = load_model('../saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
aparent_encoder = get_aparent_encoder(lib_bias=4)

#setting up files
fastaDestination = "../fastas/"
fastaNames = ["KI270747.1"]
predDestination = "../PredictionBinaries/"
strideSizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50]

#running files
for name in fastaNames:
    contigSeq = SeqIO.read(fastaDestination + name + ".fasta", "fasta")
    seq = contigSeq.seq #actual genomic sequence from the file
    print ("PREDICTING ", contigSeq.id, " with length ", len(seq))
    for stride in strideSizes:
            print ("Stride length is: ", stride)
            repPeriod = name.replace(".", "_")
            filename = predDestination + name + "Predictions/" +repPeriod + "_cutPredsStrideLen" + str(stride)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            #y=np.zeros((12,25))
            #print(y)
            x,y = find_polya_peaks_memoryFriendly(
                aparent_model,
                aparent_encoder,
                seq,
                sequence_stride=stride,
                conv_smoothing=False,
                peak_min_height=0.01,
                peak_min_distance=50,
                peak_prominence=(0.01, None),
            )
            np.save(filename, y )
    print ("FINISHED")
