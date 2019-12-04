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
import datetime
from multiprocessing import Process
import logging


from aparent.predictor import *
##################################################
#import bioPython for working with FASTA files
from Bio import SeqIO
##################################################

# Gather timing info
startTime = datetime.datetime.now()
endTime = datetime.datetime.now()
timeElapsed = endTime-startTime
print("startTime: " + str(startTime))
logging.basicConfig(filename=('predictingUnlocalizedRegionsLog_'+str(startTime)+'.log')\
,filemode='w',format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)
# created logging file.  Write logging.debug('message') to write the message to a log file

'''
#loading model
aparent_model = load_model('../saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
aparent_encoder = get_aparent_encoder(lib_bias=4)
'''

#setting up files
fastaDestination = "../fastas/"
fastaNames = ["CM000686.2"]
#fastaNames = ["FA270747.1"] # fake small fasta file based on KI270747.1
predDestination = "../PredictionBinaries/"
#strideSizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50]
strideSizes = [10]
#strideSizes = [40,50]
chunks = 500 # set number of chunks to break up DNA sample into
parallelFlag = 0 # set to 1 for doing multiprocessing, 0 for single thread
logging.debug('Logging predictingUnlocalizedRegions5.py')
logging.debug('Start time is: ' + str(startTime))
logging.debug('fastaNames are: ' + str(fastaNames))
logging.debug('strideSizes are: ' + str(strideSizes))
logging.debug('chunks is set to: ' + str(chunks))
logging.debug('parallelFlag is set to: ' + str(parallelFlag))


def logPrint(message):
    print(message)
    logging.debug(message)

def makeFilePath(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def findChunkSize(seq,chunks):
    seqLen = len(seq)
    chunkSizeFloat = seqLen/chunks
    logPrint('chunking seqLen=' + str(seqLen) + ' into chunks='+str(chunks))
    chunkSizeInt = int(chunkSizeFloat)
    diff = chunkSizeFloat - float(chunkSizeInt)
    # check if they are equal; if they are not, make the last chunk shorter
    finalChunkSizeInt = chunkSizeInt
    if (diff!=0):
        left = seqLen-(chunkSizeInt*(chunks-1))
        finalChunkSizeInt = left

    logPrint('chunkSizeFloat='+ str(chunkSizeFloat)+\
    ', chunkSizeInt=' + str(chunkSizeInt)+', finalChunkSizeInt='+str(finalChunkSizeInt))
    logPrint('Chunks Total Len = '+ str((chunkSizeInt*(chunks-1))+finalChunkSizeInt))
    return chunkSizeInt, finalChunkSizeInt

def mergePredFiles(name,chunks,stride,fileNameBase):
    return 0

def printTimingInfo():
    # Print timing info
    #print ("PREDICTED ", str(fastaNames), " with length ", len(seq))
    endTime=datetime.datetime.now()
    timeElapsed = endTime-startTime
    timeString = "Timing info:"+" timeElapsed: " + str(timeElapsed) + \
    " startTime: " + str(startTime) + " endTime: " + str(endTime)
    logPrint(timeString)

def runModel(thisSeq, stride, filename):
    aparent_model = load_model('../saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
    plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
    aparent_encoder = get_aparent_encoder(lib_bias=4)
    printTimingInfo()
    logPrint('thisSeqLen='+ str(len(thisSeq))+', running model for filename:' + filename)
    x,y = find_polya_peaks_memoryFriendly(
        aparent_model,
        aparent_encoder,
        thisSeq,
        sequence_stride=stride,
        conv_smoothing=False,
        peak_min_height=0.01,
        peak_min_distance=50,
        peak_prominence=(0.01, None),
    )
    logPrint('finished running model for filename:' + filename)
    np.save(filename, y )
    np.save((filename+'_peakixs'),x)
    logPrint("saved file: " + str(filename))
    #return x,y

if __name__ == '__main__':
#running files
    for name in fastaNames:
        contigSeq = SeqIO.read(fastaDestination + name + ".fasta", "fasta")
        seq = contigSeq.seq #actual genomic sequence from the file
        chunkSize, finalChunkSize = findChunkSize(seq,chunks)
        #chunkSize = len(seq)
        thisSeq=seq[0:chunkSize]
        finalSeq=seq[0:finalChunkSize]
        logPrint ("PREDICTING "+ str(contigSeq.id)+ " with length "+ str(len(seq)))
        for stride in strideSizes:
                logPrint ("Stride length is: " + str(stride))
                #endTime = datetime.datetime.now()
                #logPrint(endTime)
                printTimingInfo()
                repPeriod = name.replace(".", "_")
                #filename = predDestination + name + "Predictions/" +repPeriod + "_cutPredsStrideLen" + str(stride)
                fileNameBase = predDestination + name + "Predictions/" +repPeriod + "_cutPredsStrideLen" + str(stride)
                procs = []
                for chunk in range (0,chunks):
                    #p =
                    filename = fileNameBase + "_totChunks" + str(chunks) + "chunk" + str(chunk+1)
                    makeFilePath(filename)

                    if (chunk==chunks-1):
                        thisSeq = np.zeros(len(thisSeq))
                        thisSeq = seq[(chunkSize*chunk):]
                        thisSeq = np.trim_zeros(thisSeq)
                    else:
                        #logPrint(str(chunk))
                        #logPrint(str(chunkSize))
                        #logPrint(str(len(thisSeq)))
                        thisSeq = seq[(chunkSize*chunk):(chunkSize*(chunk+1))]
                        #logPrint(str(len(thisSeq)))
                    logPrint('Set chunk='+str(chunk)+' sequence to len=' +str(len(thisSeq)))
                    #y=np.zeros((12,25))
                    #print(y)
                    logPrint("starting to find polya_peaks for filename= " + str(filename))
                    if (parallelFlag == 1):
                        logPrint("spawning new process for filename= " + str(filename))
                        p = Process(target=runModel, args=(thisSeq,stride,filename,))
                        procs.append(p)
                        p.start()
                    else:
                        runModel(thisSeq, stride, filename)
                    #x,y = runModel(seq, stride, filename)
                    '''
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
                    '''
                if (parallelFlag == 1):
                    for p in procs:
                        p.join()
                if(chunks>1):
                    logPrint('Merging files')
                    mergeSuccess = mergePredFiles(name,chunks,stride,fileNameBase)
        logPrint ("FINISHED")
        printTimingInfo()
