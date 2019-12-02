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


#loading model
aparent_model = load_model('./saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
aparent_encoder = get_aparent_encoder(lib_bias=4)

#setting up files, prediction cor chr 21
fastaDestination = "./fastas/"
fastaNames = ["chrY"]
predDestination = "./PredictionBinaries/"
#strideSizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50]
strideSizes = [10]
increaseSize = 100000
#running files
for name in fastaNames:
    contigSeq = SeqIO.read(fastaDestination + name + ".fasta", "fasta")
    seq = contigSeq.seq #actual genomic sequence from the file
    #split seq into 100K portions
    print ("PREDICTING ", contigSeq.id, " with length ", len(seq))
    for stride in strideSizes:
            print ("Stride length is: ", stride)
            start = 0
            end = increaseSize - 1
            for i in range(0,int(len(seq)/stride)):
                startTime = time.time()
                sliceSeq = seq[start:end + 1]
                x,y = find_polya_peaks_memoryFriendly(
                    aparent_model,
                    aparent_encoder,
                    sliceSeq,
                    sequence_stride=stride,
                    conv_smoothing=False,
                    peak_min_height=0.01,
                    peak_min_distance=50,
                    peak_prominence=(0.01, None),
                )
                repPeriod = name.replace(".", "_")
                np.save(predDestination + name + "Predictions/" +repPeriod + "_StrideLen" + str(stride) + "Start" + str(start+ 1) + "End" + str(end + 1), y )
                secondsDiffs = time.time()-startTime
                print ("Time for ",start, "to", end, ":", str(int(secondsDiffs/60)) + " mins " + str(secondsDiffs%60.0))
                start += increaseSize
                end += increaseSize
            restSeq = seq[end:]
            x,y = find_polya_peaks_memoryFriendly(
                aparent_model,
                aparent_encoder,
                restSeq,
                sequence_stride=stride,
                conv_smoothing=False,
                peak_min_height=0.01,
                peak_min_distance=50,
                peak_prominence=(0.01, None),
            )
            repPeriod = name.replace(".", "_")
            np.save(predDestination + name + "Predictions/" +repPeriod + "_StrideLen" + str(stride) + "Start" + str(end + 1) + "End" + str(len(seq)), y )
    print ("FINISHED")
