# predictingChromosomes.py - cleaned up file to predict chromosomes in slices and properly merge them back together

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

# Gather timing info
startTime = datetime.datetime.now()
logTime = startTime
endTime = datetime.datetime.now()
timeElapsed = endTime-startTime
print("startTime: " + str(startTime))
curPath = os.getcwd()
logFileName = 'predictingChromosomesLog_'+str(logTime)+'.log'
logFilePath = './' + logFileName
#logFilePath = os.path.join(curPath,logFileName)
logFilePath = curPath + logFileName
logging.basicConfig(filename=logFileName,filemode='w',format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)

def logPrint(message):
    print(message)
    logging.debug(message)

#loading model
aparent_model = load_model('../saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
aparent_encoder = get_aparent_encoder(lib_bias=4)

#setting up files / configuration
fastaDestination = "../fastas/"
#fastaNames = ["FA270747.1"] # fake small fasta file based on KI270747.1
fastaNames = ["CM000677.2", "CM000681.2", "CM000686.2", "CM000663.2", "CM000676.2", "CM000675.2"]
predDestination = "../PredictionBinaries/"
strideSizes = [50]
sliceSize = 100000
#sliceSize = 1000
padSize = 400 # arbitrarily chosen large pad size to make sure we have enough room at end
fileSize = sliceSize + padSize*2 # file size before merging; put one pad on each size which we can slice off at end
parallelFlag = 0 # set to 1 for doing multiprocessing, 0 for single thread
mergeFlag = 1 # 0 for skip merge, 1 for predict and merge, 2 for just merge
reverseCompFlag = 1 # 0 for processing sequence in order saved in fasta file, 1 for flipping to reverse complement

# logging initial details about run
logPrint('Logging ' + os.path.basename(__file__)) #predictingChromosomes.py')
logPrint('Start time is: ' + str(startTime))
logPrint('fastaNames are: ' + str(fastaNames))
logPrint('strideSizes are: ' + str(strideSizes))
logPrint('fileSize: ' + str(fileSize) + ', sliceSize: ' + str(sliceSize) + ', padSize: ' + str(padSize))
logPrint('mergeFlag is set to: ' + str(mergeFlag))
logPrint('parallelFlag is set to: ' + str(parallelFlag))
logPrint('reverseCompFlag is set to: ' + str(reverseCompFlag))

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

def getSliceNum(filepath):
    baseStr, str1 = filepath.split('SliceNum')
    sliceNumStr, str2 = str1.split('Start')
    sliceNum = int(sliceNumStr)
    return sliceNum

def getFileStats(filepath):
    baseStr, str1 = filepath.split('SliceNum')
    name,strideStr = baseStr.split('_cutPredsStrideLen')
    stride = int(strideStr)
    sliceNumStr, str2 = str1.split('Start')
    sliceNum = int(sliceNumStr)
    startStr, str3 = str2.split('End')
    endStr, str4 = str3.split('TotLen')
    lenStr, str5 = str4.split('.n')
    start = int(startStr)
    end = int(endStr)
    totLen = int(lenStr)
    #return sliceNum, start, end
    return sliceNum, start, end, stride, name, totLen

def mergePredFiles(name,chunks,stride,fileNameBase):
    return 0

def getMergeList(stride, directory):
    mergeList = { }
    filepathList = { }
    logPrint('getting merge list from dir:' + directory + ', starting mergeList:')
    logPrint(mergeList)
    for entry in os.scandir(directory):
        filepath = entry.path
        if not ('peakixs' in filepath):
            if('SliceNum' in filepath):
                sliceNum = getSliceNum(filepath)
                appendArray = np.load(filepath)
                mergeList[sliceNum] = appendArray
                filepathList[sliceNum] = filepath
    logPrint('mergeList created from dir:'+ directory+', ending mergeList:')
    logPrint(len(mergeList))
    return mergeList, filepathList

def mergeSlicedFiles(fileNameBase,mergeList,filepathList,totLen = 0):
    spliceDiff = padSize # amount of bps (206) to cut off beginning and end of arrays when merging
    mergeArray1 = mergeList[0][0]
    #mergeArray1 = mergeArray1[0]
    mergeFilepath = filepathList[0]
    sliceNum, start, end, stride, name, totLen = getFileStats(mergeFilepath)
    firstEnd = end # also the slice size
    y = np.empty(totLen)
    prevEnd = 0
    mergeStart = 0
    # do something special to start for the 0th file
    mergeArray0_lastInd = firstEnd-spliceDiff
    mergeArray0 = mergeArray1[:mergeArray0_lastInd+1]
    y[:mergeArray0_lastInd+1] = mergeArray0
    logPrint('Added to y between ' + str(0) + ':' + str(mergeArray0_lastInd) + ' with filename='+str(mergeFilepath))
    logPrint('mergeArray0[0]='+str(mergeArray0[0])+', mergeArray0[1]='+str(mergeArray0[1])+\
    ', mergeArray0['+str(mergeArray0_lastInd-1)+']='+str(mergeArray0[mergeArray0_lastInd-1])+', mergeArray0['+str(mergeArray0_lastInd)+']='+str(mergeArray0[mergeArray0_lastInd]))
    logPrint('     y['+str(0)+']='+str(y[0])+',      y['+str(0+1)+']='+str(y[0+1])+\
    ',         y['+str(mergeArray0_lastInd-1)+']='+str(y[mergeArray0_lastInd-1])+',        y['+str(mergeArray0_lastInd)+']='+str(y[mergeArray0_lastInd]))

    # add arrays from each file generated, then merge
    for i in range(1,len(mergeList)):
        mergeArray1 = mergeList[i][0] # full array + 206 at each end as padding
        mergeFilepath = filepathList[i]
        curSliceNum, curStart, curEnd, stride2, name2, totLen2 = getFileStats(mergeFilepath) # cur- are stats from filename
        #calculate the actual start and end of our splice zones in terms of indexes into the orig sample
        mergeStart = curStart + spliceDiff # the actual start of this useful data is one splice
        mergeEnd = curEnd - spliceDiff
        arrayLen = len(mergeArray1)
        mergeLen = arrayLen - spliceDiff*2
        # calculate and append the part of the array we want to append
        mergeArray = mergeArray1[(spliceDiff):(mergeLen+spliceDiff)] # TODO: this +1 may be problematic- this error may be coming from our prediction code in creating the files
        y[mergeStart:mergeEnd+1] = mergeArray
        # print extra debug info afterwards
        '''
        logPrint('Added to y between ' + str(mergeStart) + ':' + str(mergeEnd) + ' with filename='+str(mergeFilepath))
        logPrint('mergeArray[0]='+str(mergeArray[0])+', mergeArray[1]='+str(mergeArray[1])+\
        ', mergeArray['+str(mergeLen-2)+']='+str(mergeArray[mergeLen-2])+', mergeArray['+str(mergeLen-1)+']='+str(mergeArray[mergeLen-1]))
        logPrint('     y['+str(mergeStart)+']='+str(y[mergeStart])+',      y['+str(mergeStart+1)+']='+str(y[mergeStart+1])+\
        ',         y['+str(mergeEnd-1)+']='+str(y[mergeEnd-1])+',        y['+str(mergeEnd)+']='+str(y[mergeEnd]))
        '''
        prevEnd = mergeEnd
    # save end of final slice to finish off the merge
    y[mergeStart:] = mergeArray1[spliceDiff:]
    filename = fileNameBase + 'SliceSize' +str(firstEnd)+'TotLen' +str(totLen) + '_merged'
    logPrint('Saving merged file with len(y)='+str(len(y))+' and y= ')
    logPrint(y)
    saveFile([0,0,0],y,filename)
    return y

def printTimingInfo():
    # Print timing info
    #print ("PREDICTED ", str(fastaNames), " with length ", len(seq))
    endTime=datetime.datetime.now()
    timeElapsed = endTime-startTime
    timeString = "Timing info:"+" timeElapsed: " + str(timeElapsed) + \
    " startTime: " + str(startTime) + " endTime: " + str(endTime)
    logPrint(timeString)

# runModel function for running in new thread asynchronously (not used currently)
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

#def saveFile(x, y, filename, mergeList, filenameList):
def saveFile(x, y, filename):
    makeFilePath(filename)
    np.save(filename,y)
    np.save((filename+'_peakixs'),x)
    logPrint("saved file: " + str(filename))

if __name__ == '__main__':
#running files
    for name in fastaNames:
        contigSeq = SeqIO.read(fastaDestination + name + ".fasta", "fasta")
        seq = contigSeq.seq
        if (reverseCompFlag == 1):
            logPrint('Reverse Complement is being used.  Reversing seq...')
            seq = contigSeq.seq.reverse_complement()
        logPrint ("PREDICTING "+ str(contigSeq.id)+ " with length "+ str(len(seq)))
        for stride in strideSizes:
            logPrint ("Stride length is: " + str(stride))
            repPeriod = name.replace(".", "_")
            dirDest = predDestination + str(logTime) + '/'
            makeFilePath(dirDest)
            predsAdd = "Predictions/"
            if(reverseCompFlag==1):
                predsAdd = "Predictions_revComp/"
            directory = predDestination  + name + predsAdd
            fileNameBase = directory+ repPeriod + "_cutPredsStrideLen" + str(stride)
            #fileNameBase = os.path.join(directory, fileNameBase1, str(startTime))
            mergeList = { }
            filenameList = { }
            if (mergeFlag<2):
                start = 0
                end = fileSize - 1
                sliceNum = 0
                printTimingInfo()
                numSlices = len(seq)/sliceSize
                for i in range(0,int(len(seq)/stride)):
                    logPrint('Processing to seqInd:' + str(end) + ' out of total len:' + str(len(seq)) + \
                    ', remaining bps:' + str(len(seq)-end) + ', %:' + str(100*(end/len(seq))))
                    if(end>len(seq)):
                        end = len(seq)-1
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

                    filename = fileNameBase + "SliceNum" +str(sliceNum)+"Start" + str(start) + "End" + str(end) +"TotLen"+str(len(seq))
                    saveFile(x,y,filename)
                    if(end==len(seq)-1):
                        logPrint('Breaking out of for loop; end='+str(end)+', len(seq)='+str(len(seq)))
                        break
                    sliceNum +=1
                    start += fileSize - padSize*2
                    end += fileSize - padSize*2
                    #logPrint('Processed section for filename: ' + filename)
                    printTimingInfo()
            else:
                logPrint('Skipping predictions b/c mergeFlag set to:' + str(mergeFlag))
            if(mergeFlag > 0):
                logPrint('Merging files, mergeFlag='+str(mergeFlag)+', mergeListLen='+str(len(mergeList)))
                #mergeSuccess = mergePredFiles(name,chunks,stride,fileNameBase)
                mergeList, filenameList = getMergeList(stride,directory)
                logPrint(len(mergeList))
                mergeSuccess = mergeSlicedFiles(fileNameBase,mergeList,filenameList)
            # multiprocessing style of running the model
            if (parallelFlag==1):
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
            #if(chunks>1):
            # merge if mergeFlag = 1 for run and merge or mergeFlag=2 for just merge

        logPrint ("FINISHED")
        printTimingInfo()
