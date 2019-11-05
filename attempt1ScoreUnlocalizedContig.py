#attempt number1 of getting APARENT to score a long contig 
# 10/29/19 
#AML

#imports from aparent_example_pas_detection
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

from aparent.predictor import *
##################################################
#import bioPython for working with FASTA files
from Bio import SeqIO
##################################################

#open the FASTA for GL000219.1 (https://www.ncbi.nlm.nih.gov/nuccore/GL000219.1?report=fasta) unplaced human genomic assembly

#bioPython instructions http://biopython.org/DIST/docs/tutorial/Tutorial.html 
contigSeq = SeqIO.read("GL000219.1.fasta", "fasta")
seq = contigSeq.seq #actual genomic sequence from the file

#loading model
aparent_model = load_model('./saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
aparent_encoder = get_aparent_encoder(lib_bias=4)

#detecting peaks
peak_ixs, polya_profile = find_polya_peaks(
    aparent_model,
    aparent_encoder,
    seq,
    sequence_stride=5,
    conv_smoothing=True,
    peak_min_height=0.01,
    peak_min_distance=50,
    peak_prominence=(0.01, None)
)

print("Peak positions = " + str(peak_ixs))
print("PolyA profile shape = " + str(polya_profile.shape))

