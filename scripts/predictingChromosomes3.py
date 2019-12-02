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
def logPrint(message):
    print(message)
    logging.debug(message)

#loading model
aparent_model = load_model('../saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5')
plot_model(aparent_model, show_shapes = True, to_file='APARENTmodel.png')
aparent_encoder = get_aparent_encoder(lib_bias=4)


#setting up files
fastaDestination = "../fastas/"
#fastaNames = ["CM000666.2"]
#fastaNames = ["FA270747.1"] # fake small fasta file based on KI270747.1
fastaNames = ["CM000665.2"]
predDestination = "../PredictionBinaries/"
#strideSizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50]
strideSizes = [10]
#strideSizes = [40,50]
#chunks = 10 # set number of chunks to break up DNA sample into
sliceSize = 100000
#sliceSize = 10000
padSize = 186+(strideSizes[0]*2) # pad for slicing off at end (206)
increaseSize = sliceSize + padSize*2 # put one pad on each size which we can slice off at end
parallelFlag = 0 # set to 1 for doing multiprocessing, 0 for single thread
mergeFlag = 1 # 0 for skip merge, 1 for predict and merge, 2 for just merge
#logLoc = logging.getLoggerClass().root.handlers[0].baseFilename
'''
logfilenames = []
for handler in logger.handlers:
    try:
        logfilenames.append(handler.fh.name)
    except:
        pass
'''
#logPrint('Log location: ' + str(logLoc))
#print('Log location: ' + str(logfilenames))
logPrint('Logging predictingChromosomes3.py')
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

# predDestination = "../PredictionBinaries/"
# name = CM000686.2
# repPeriod = CM000686_2
# fileNameBase = predDestination  + name + "Predictions/" +repPeriod + "_cutPredsStrideLen" + str(stride)
# directory: Z:\WindowsFolders\repos\aparentGenomeTesting\PredictionBinaries\CM000686.2Predictions
# filename: CM000686_2_cutPredsStrideLen10TotLen57227415SliceNum520Start52107121End52207532.npy
# directory needs to be: ..\PredictionBinaries\CM000686.2Predictions
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
                #logPrint('Added SliceNum:' + str(sliceNum))
    #directory = fileNameBase
    logPrint('mergeList created from dir:'+ directory+', ending mergeList:')
    #logPrint(mergeList)
    logPrint(len(mergeList))
    return mergeList, filepathList

def mergeSlicedFiles(fileNameBase,mergeList,filepathList,totLen = 0):

    spliceDiff = padSize # amount of bps (206) to cut off beginning and end of arrays when merging
    mergeArray1 = mergeList[0][0]
    mergeArray1 = mergeArray1[0]
    mergeFilepath = filepathList[0]
    sliceNum, start, end, stride, name, totLen = getFileStats(mergeFilepath)
    firstEnd = end # also the slice size
    y = np.empty(totLen) # could make this more efficient by preloading the length
    prevEnd = 0
    mergeStart = 0

    for i in range(0,len(mergeList)):
    #for i in range(0,10):
        mergeArray1 = mergeList[i][0] # full array + 206 at each end as padding
        #logPrint(mergeArray1)
        mergeFilepath = filepathList[i]
        curSliceNum, curStart, curEnd, stride2, name2, totLen2 = getFileStats(mergeFilepath) # cur- are stats from filename
        #logPrint('curSliceNum:' + str(curSliceNum) + ', curStart:'+str(curStart)+', curEnd:'+str(curEnd) + ', mergeArray1len:'+str(len(mergeArray1)))

        #calculate the actual start and end of our splice zones in terms of indexes into the orig sample
        mergeStart = curStart + spliceDiff # the actual start of this useful data is one splice
        mergeEnd = curEnd - spliceDiff


        arrayLen = len(mergeArray1)
        mergeLen = arrayLen - spliceDiff*2
        #logPrint('mergeStart:'+str(mergeStart)+', mergeEnd:'+str(mergeEnd)+', arrayLen:'+str(arrayLen)+', mergeLen:'+str(mergeLen))

        # calculate and append the part of the array we want to append
        mergeArray = mergeArray1[spliceDiff:(mergeLen+spliceDiff)]
        np.append(y,mergeArray)

        # print extra debug info afterwards
        #logPrint('Appended array from file:'+str(mergeFilepath))
        #logPrint('len(mergeArray)='+str(len(mergeArray))+', mergeArray=')
        #logPrint(mergeArray)
        #logPrint('len(y)='+str(len(y))+', y=')
        #logPrint(y)
        #logPrint( ', SliceNum='+str(curSliceNum)+\
        #', start='+str(mergeStart)+', end='+str(mergeEnd)+', mergeArrayLen:' + str(len(mergeArray)))

        prevEnd = mergeEnd
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
    #sliceNum = getSliceNum(filename)
    #mergeList[sliceNum] = y
    #filenameList[sliceNum] = filename

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
            directory = predDestination  + name + "Predictions/"
            fileNameBase = directory +repPeriod + "_cutPredsStrideLen" + str(stride)
            mergeList = { }
            filenameList = { }
            if (mergeFlag<2):
                start = 0
                end = increaseSize - 1
                sliceNum = 0
                #endTime = datetime.datetime.now()
                #logPrint(endTime)
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

                    #np.save(predDestination + name + "Predictions/" +repPeriod + "_StrideLen" + str(stride) + "SliceNum" +str(sliceNum)+"Start" + str(start+ 1) + "End" + str(end + 1), y )
                    filename = fileNameBase + "SliceNum" +str(sliceNum)+"Start" + str(start) + "End" + str(end) +"TotLen"+str(len(seq))
                    #saveFile(x,y,filename, mergeList, filenameList)
                    saveFile(x,y,filename)
                    if(end==len(seq)-1):
                        logPrint('Breaking out of for loop; end='+str(end)+', len(seq)='+str(len(seq)))
                        break
                    sliceNum +=1
                    start += increaseSize - padSize*2
                    end += increaseSize - padSize*2
                    #logPrint('Processed section for filename: ' + filename)
                    printTimingInfo()
                '''
                restSeq = seq[start:]
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
                filename = fileNameBase + "SliceNum" +str(sliceNum)+"Start" + str(start) + "End" + str(end) +"TotLen"+str(len(seq))
                saveFile(x,y,filename, mergeList, filenameList)
                '''
            else:
                logPrint('Skipping predictions b/c mergeFlag set to:' + str(mergeFlag))
            if(mergeFlag > 0):
                logPrint('Merging files, mergeFlag='+str(mergeFlag)+', mergeListLen='+str(len(mergeList)))
                #mergeSuccess = mergePredFiles(name,chunks,stride,fileNameBase)
                mergeList, filenameList = getMergeList(stride,directory)
                logPrint(len(mergeList))
                mergeSuccess = mergeSlicedFiles(fileNameBase,mergeList,filenameList)
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

        logPrint ("FINISHED")
        printTimingInfo()
