import numpy as np
from matplotlib import pyplot as plt


filePath = '../PredictionBinaries/'
fastaNames = ["KI270737.1"]
#fastaName2 = fastaName.replace(".", "_")
fileAdd = 'KI270737_1_cutPredsStrideLen1.npy'
strideSizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50]
likelyThreshold = .15 # global min threshold to use for predicting PAS regions within a stride
agreeThreshold = 5    # global min threshold to use for predicting agreement between stride sizes for PAS region
#fullData = np.zeros((,len(np.load)))
lenData = 5
maxIndex = 0


# function to replace periods
def repPeriod(name):
    return name.replace(".", "_")

def fillData():
    global maxIndex
    fullPath = filePath + fastaNames[0] + 'Predictions/' + repPeriod(fastaNames[0]) +"_cutPredsStrideLen" + '1' + '.npy'
    data = np.load(fullPath)
    lenData = len(data[0,:])
    print('lenData set to: ' + str(lenData))
    #print(len(data[0,:]))
    #print(len(data[:,0]))
    #print(data[0,:])
    #fullData = np.empty(((len(strideSizes)+1),lenData))
    fullData = np.empty(((max(strideSizes)+2),lenData))
    maxIndex = len(fullData[:,0])-1
    print('maxIndex=' + str(maxIndex))
    return fullData

# plot a single graph with
def plot_data(data, stride):
    indices = np.array(list(range(len(data))))
    print(len(data))
    data2 = data[data>.15]
    indices2 = indices[data>.15]
    print(indices2)
    print(data2)
    #print(indices)
    #plt.plot(indices,data,linestyle="",marker=".")
    plt.plot(indices2,data2,linestyle="",marker=".")
    plt.gca().set(title=('APA Likelihood for stride = ' + str(stride)),ylabel='Likelihood',xlabel='Gene Index')
    plt.show()

# plot a single graph with all
# data is an array with data[0,:] as the indices and data[1:x,:] as the values
def plotThresholds(data,fastaName):
    print('Plotting thresholds for '+fastaName+ ' ...')
    indices = data[0,:]
    print('indices: ')
    print(indices)
    print('data[:,1]: ')
    print(data[:,1])
    #for array in data[:,0]
    #for i in range(1,len(data[:,0])):
    for stride in strideSizes:
        try:
            plt.plot(data[0,:],data[stride,:],linestyle="",marker=".",label=stride)
            #plt.legend().set(loc='right')
            #print('Plotting row ' + str(stride) + ':')
            #print(data[stride,:])
        except:
            #print('Skipping stride '+ str(stride))
            four = 4

        #if (data[stride,1]!=np.nan):
            #break
        #plt.plot(data[0,:],data[stride,:],linestyle="",marker=".")

    #plt.plot(indices,data[1,:],linestyle="",marker=".")
    plt.gca().set(title=('APA Likelihood for each Marker for '+fastaName),ylabel='Likelihood',xlabel='Gene Index')
    plt.legend(loc='upper right')
    plt.legend(bbox_to_anchor=(1.0,1.0))
    plt.gcf().set_size_inches(8,8)

    plt.show()
    return 0

# calculate above a certain
def calcThresholds(fullData, data, threshold):
    print('Calculating thresholds ...')
    #print(data[0,:])
    indices = fullData[0,:]
    newIndices = indices[data[0,:]>likelyThreshold]
    tempData = data[data>likelyThreshold]
    #print(len(newIndices))
    #print(len(tempData))
    newData = np.array([[newIndices],[tempData]])
    #print(newData)

    #newData = data[data>threshold]
    #newData[0,:] = indices

    #print(len(newData[0,:]))
    #for i in range(0,len(data)):
    #    if newData[0,i] < threshold:
    #        newData[i] = 0
    return newData
    #return data[data>threshold]
    #return data
    #return 0

# main function to run thresholding and plot thresholds
def runThresholding():
    global threshold
    global lenData
    global fullData
    #print(len(fullData[0,:]))
    #print(len(fullData[:,0]))
    #fullData[0,:] = np.load(fullPath)
    fullData[0,:] = np.arange(0,len(fullData[0,:]))
    print('fullData[0,:] set to: ')
    print(fullData[0,:])
    for fastaName in fastaNames:
        #for stride in range(0,51):
        #for stride in range(1,2):
        for stride in strideSizes:
            #print('starting stride ' + str(stride))
            # file is named: KI270737_1_cutPredsStrideLen1.npy
            fullPath = filePath + fastaName + 'Predictions/' + repPeriod(fastaName) +"_cutPredsStrideLen" + str(stride) + '.npy'
            data = np.load(fullPath)
            #print(data)
            threshData = calcThresholds(fullData, data, likelyThreshold)
            #print('threshData: ')
            #print(threshData)
            #print(len(threshData[0,0,:]))
            #for arr in threshData[0,0,:]:
            for i in range(0,len(threshData[0,0,:])):
                #print(threshData[0,0,i])
                # set the fullData location for stride to the thresholded value
                fullData[stride,int(threshData[0,0,i])] = threshData[1,0,i]
                # add 1 to the summing row at the bottom of the matrix for future checking agreement
                fullData[maxIndex,int(threshData[0,0,i])] +=1
            #    print('Placed val=' + str(threshData[1,0,i]) + ' at fullData['\
            #    + str(stride) +','+str(threshData[0,0,i])+']')
            #print(fullData[stride,:])
            # convert all the zeros to nan
            fullData[stride,(fullData[stride,:]==0)] = np.nan
            #fullData[stride,:] = threshData
            #plot_data(data[0,:],stride)
            # chance to stop the loop with a setting:
            if (stride>50):
                break
    #calcThresholds(, threshold
    #print(np.size(fullData))
    #print(fullData[0,:])
    #print(fullData[:,0])
    #print(fullData)

# check if more than one stride size says that a PAS site is there
def checkAgreement(fullData):
    global maxIndex
    print('Checking agreement between stride sizes ...')
    agreeThresholds = fullData[0,fullData[maxIndex,:]>agreeThreshold]
    print(agreeThresholds)
    print(len(agreeThresholds))
    #print('maxIndex=' +str(maxIndex)) # =51
    #print(len(fullData[:,0])) =52
    print(fullData[(len(fullData[:,0])-1),:])
    return agreeThresholds


def printResults(fullData,fastaName):
    print('Reults of Thresholding:')
    for stride in strideSizes:
        threshResults = fullData[stride,fullData[stride,:]>likelyThreshold]
        threshIndices = fullData[0,fullData[stride,:]>likelyThreshold]
        print('For fasta=' + fastaName+ \
        ', using stride length=' + str(stride) + ', there are '\
        + str(len(threshResults)) + ' strong results, and '\
        + 'the values over threshold=' + str(likelyThreshold) + ' are:')
        print(threshResults)
        print('with indices =')
        print(threshIndices)



fullData = fillData()
runThresholding()
printResults(fullData,fastaNames[0])
strongThresholds = checkAgreement(fullData)
plotThresholds(fullData,fastaNames[0])
