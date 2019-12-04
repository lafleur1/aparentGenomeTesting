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
from imp import reload


from aparent.predictor import *
##################################################
#import bioPython for working with FASTA files
from Bio import SeqIO
##################################################

def makeFilePath(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #if not os.path.exists(filename):
    #    os.makedirs(filename)

def logPrint(message):
    print(message)
    logging.debug(message)

# Gather timing info
startTime = datetime.datetime.now()
logTime = startTime
endTime = datetime.datetime.now()
timeElapsed = endTime-startTime
print("startTime: " + str(startTime))
curPath = os.getcwd()
logFileName = 'predictingChromosomesLog_'+str(logTime)+'.log'
logFilePath = './' + logFileName
logFilePath = curPath+logFileName
#makeFilePath(logFilePath)
#makeFilePath('./')
#reload(logging)
#for handler in logging.root.handlers[:]:
#    logging.root.removeHandler(handler)
logging.basicConfig(filename=logFileName,filemode='w',format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)
#logger = logging.getLogger()
#logging.basicConfig(filemode='w',format='%(name)s - %(levelname)s - %(message)s',level=logPrint)
#makeFilePath('./')
#logging.basicConfig(filename=('predictingChromosomesLog_'+str(logTime)+'.log')\
#,filemode='w',format='%(name)s - %(levelname)s - %(message)s',level=logPrint)
# created logging file.  Write logPrint('message') to write the message to a log file


#loading model
aparent_model = load_model('../saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
aparent_encoder = get_aparent_encoder(lib_bias=4)


#setting up files
fastaDestination = "../fastas/"
fastaNames = ["CM000686.2"]
#fastaNames = ["FA270747.1"] # fake small fasta file based on KI270747.1
predDestination = "../PredictionBinaries/"
#strideSizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50]
strideSizes = [10]
#strideSizes = [40,50]
#chunks = 10 # set number of chunks to break up DNA sample into
sliceSize = 100000
#sliceSize = 10000
padSize = 186+(strideSizes[0]*2) # pad for slicing off at end
increaseSize = sliceSize + padSize*2 # put one pad on each size which we can slice off at end
parallelFlag = 0 # set to 1 for doing multiprocessing, 0 for single thread
mergeFlag = 0 # 0 for skip merge, 1 for predict and merge, 2 for just merge
logLoc = logging.getLoggerClass().root.handlers[0].baseFilename
'''
logfilenames = []
for handler in logger.handlers:
    try:
        logfilenames.append(handler.fh.name)
    except:
        pass
'''
logPrint('Log location: ' + str(logLoc))
#print('Log location: ' + str(logfilenames))
logPrint('Logging predictingChromosomes.py')
logPrint('Start time is: ' + str(startTime))
logPrint('fastaNames are: ' + str(fastaNames))
logPrint('strideSizes are: ' + str(strideSizes))
#logPrint('chunks is set to: ' + str(chunks))
logPrint('increaseSize: ' + str(increaseSize) + ', sliceSize: ' + str(sliceSize) + ', padSize: ' + str(padSize))
logPrint('mergeFlag is set to: ' + str(mergeFlag))
logPrint('parallelFlag is set to: ' + str(parallelFlag))




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

def mergeSlicedFiles(name,chunks,stride,fileNameBase):
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
    '''
    aparent_model = load_model('../saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
    plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
    aparent_encoder = get_aparent_encoder(lib_bias=4)
    '''
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

def saveFile(x, y, filename):
    #makeFilePath(filename)
    np.save(filename,y)
    np.save((filename+'_peakixs'),x)
    logPrint("saved file: " + str(filename))

if __name__ == '__main__':
#running files
    for name in fastaNames:
        contigSeq = SeqIO.read(fastaDestination + name + ".fasta", "fasta")
        seq = contigSeq.seq #actual genomic sequence from the file
        #chunkSize, finalChunkSize = findChunkSize(seq,chunks)
        #chunkSize = len(seq)
        #thisSeq=seq[0:chunkSize]
        #finalSeq=seq[0:finalChunkSize]
        logPrint ("PREDICTING "+ str(contigSeq.id)+ " with length "+ str(len(seq)))
        for stride in strideSizes:
                logPrint ("Stride length is: " + str(stride))
                repPeriod = name.replace(".", "_")
                #filename = predDestination + name + "Predictions/" +repPeriod + "_cutPredsStrideLen" + str(stride)
                dirDest = predDestination + '/' + str(logTime)
                makeFilePath(dirDest)
                fileNameBase = predDestination  + name + "Predictions/" +repPeriod + "_cutPredsStrideLen" + str(stride)
                if (mergeFlag<2):
                    start = 0
                    end = increaseSize - 1
                    sliceNum = 0
                    #endTime = datetime.datetime.now()
                    #logPrint(endTime)
                    printTimingInfo()
                    for i in range(0,int(len(seq)/stride)):
                        print('Processing to seqInd:' + str(end) + ' out of total len:' + str(len(seq)) + \
                        ', remaining bps:' + str(len(seq)-end) + ', %:' + str(100*(end/len(seq))))
                        sliceSeq = seq[start:end+1]

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

                        #np.save(predDestination + name + "Predictions/" +repPeriod + "_StrideLen" + str(stride) + "SliceNum" +str(sliceNum)+"Start" + str(start+ 1) + "End" + str(end + 1), y )
                        filename = fileNameBase +"TotLen"+str(len(seq))+ "SliceNum" +str(sliceNum)+"Start" + str(start+ 1) + "End" + str(end + 1)
                        saveFile(x,y,filename)
                        sliceNum +=1
                        start += increaseSize - padSize
                        end += increaseSize - padSize
                        #logPrint('Processed section for filename: ' + filename)
                        printTimingInfo()
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
                    #np.save(predDestination + name + "Predictions/" +repPeriod + "_StrideLen" + str(stride) + "SliceNum" +str(sliceNum)+"Start" + str(end+ 1) + "End" + str(len(seq)), y )
                    filename = fileNameBase + "SliceNum" +str(sliceNum)+"Start" + str(start+ 1) + "End" + str(end + 1)
                    saveFile(x,y,filename)
                '''
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

                if (parallelFlag == 1):
                    for p in procs:
                        p.join()
                '''
                #if(chunks>1):
                # merge if mergeFlag = 1 for run and merge or mergeFlag=2 for just merge
                if(mergeFlag > 0):
                    logPrint('Merging files')
                    #mergeSuccess = mergePredFiles(name,chunks,stride,fileNameBase)
                    mergeSuccess = mergeSlicedFiles(name,stride,fileNameBase)
        logPrint ("FINISHED")
        printTimingInfo()
