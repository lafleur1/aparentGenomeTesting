# compare merged vs full run for FAsdfsd

import numpy as np
from matplotlib import pyplot as plt

directory = '../PredictionBinaries/FA270747.1Predictions/compare/'
#directory = '../PredictionBinaries/KK270747.1Predictions/compare/'

# master is filename1
#filename1 = 'FA270747_1_cutPredsStrideLen10SliceSize35209TotLen35210_merged.npy'
filename1 = 'FA270747_1_cutPredsStrideLen50SliceNum0Start0End35209TotLen35210.npy'
filename2 = 'FA270747_1_cutPredsStrideLen50SliceSize1571TotLen35210_merged.npy'
filename3 = 'FA270747_1_cutPredsStrideLen50SliceNum0Start0End1571TotLen35210.npy'
filename4 = 'FA270747_1_cutPredsStrideLen50SliceNum1Start1000End2571TotLen35210.npy'
filename5 = 'FA270747_1_cutPredsStrideLen50SliceNum2Start2000End3571TotLen35210.npy'
#filename1 = 'KI270747_1_cutPredsStrideLen10SliceSize198734TotLen198735_merged.npy'
#filename2 = 'KK270747_1_cutPredsStrideLen10SliceSize100411TotLen198735_merged.npy'

filepath1 = directory+filename1
filepath2 = directory+filename2
filepath3 = directory+filename3
filepath4 = directory+filename4
filepath5 = directory+filename5

padSize = 206
diffSize = 572 # difference between slice size and range in file
sliceSize = 1000
region1 = np.arange(0,10)
region2 = np.arange(206,216)
region2 = np.arange(202,212)
region3 = np.arange(0,sliceSize+diffSize) # beginning for single/first slice
region4 = np.arange(0,(sliceSize-padSize)) # first slice minus pad
region5 = np.arange(sliceSize,sliceSize*2+diffSize) # second slice
region6 = np.arange(0,sliceSize*2+diffSize) # first two slices combined (theoretically)
region7 = np.arange(sliceSize*2,sliceSize*3+diffSize) # third slice

array1 = np.load(filepath1)[0][region7]
#array1 = np.load(filepath1)
#array2 = np.load(filepath2)[region6]
array2 = np.load(filepath5)[0][region3] # load region 3 because this is a single slice
array1unpad = array1[padSize:(len(array1)-padSize)] # remove pads from both arrays
array2unpad = array2[padSize:(len(array2)-padSize)]
#array1[np.abs(array1)<.001] = 0
#array2[np.abs(array2)<.001] = 0

print('array1=')
print(array1)
print(len(array1))
print(array1unpad)
print(len(array1unpad))
print('array2=')
print(array2)
print(len(array2))
print(array2unpad)
print(len(array2unpad))

diff = array1-array2
diff2 = array1unpad - array2unpad

print('diff:')
print(sum(diff))
print(diff)
print('diff2:')
print(sum(diff2))
print(diff2)

print('debugging:')

print(region1)
diff01 = array1[region1] - array2[region1]
print('array1['+str(region1)+']: ')
print(array1[region1])
print('array2['+str(region1)+']: ')
print(array2[region1])
print('diff01['+str(region1)+']: ')
print(diff01)


print(region2)
diff01 = array1[region1] - array2[region2]
print('array1['+str(region1)+']: ')
print(array1[region1])
print('array2['+str(region2)+']: ')
print(array2[region2])
print('diff01[diff regions, eg:'+str(region2)+']: ')
print(diff01)


print(region3)
diff01 = array1[region1] - array2[region2]
print('array1['+str(region1)+']: ')
print(array1[region1])
print('array2['+str(region2)+']: ')
print(array2[region2])
print('diff01[diff regions, eg:'+str(region2)+']: ')
print(diff01)

# plot arrays and diffs with pads
indices = np.arange(0,len(array1))
plt.plot(indices,array1, color='red')
plt.plot(indices,array2, color='orange')
plt.plot(indices,diff,   color='blue')
plt.gca().set(title='APA Likelihood at bp indicess for arrays incl pads',ylabel='APA Likelihood',xlabel='Base Pair Index')
plt.show()

# plot arrays and diffs unpadded
indices = np.arange(0,len(array1unpad))
plt.plot(indices,array1unpad, color='red')
plt.plot(indices,array2unpad, color='orange')
plt.plot(indices,diff2,   color='blue')
plt.gca().set(title='APA Likelihood at bp indicess for arrays after removing pads',ylabel='APA Likelihood',xlabel='Base Pair Index')
plt.show()
