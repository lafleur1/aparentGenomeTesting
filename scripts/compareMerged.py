# compare merged vs full run for FAsdfsd

import numpy as np
from matplotlib import pyplot as plt

#directory = '../PredictionBinaries/FA270747.1Predictions/compare/'
directory = '../PredictionBinaries/KK270747.1Predictions/compare/'

# master is filename1
#filename1 = 'FA270747_1_cutPredsStrideLen10SliceSize35209TotLen35210_merged.npy'
#filename1 = 'FA270747_1_cutPredsStrideLen10SliceNum0Start0End35209TotLen35210.npy'
#filename2 = 'FA270747_1_cutPredsStrideLen10SliceSize10411TotLen35210_merged.npy'
filename1 = 'KI270747_1_cutPredsStrideLen10SliceSize198734TotLen198735_merged.npy'
filename2 = 'KK270747_1_cutPredsStrideLen10SliceSize100411TotLen198735_merged.npy'

filepath1 = directory+filename1
filepath2 = directory+filename2

#array1 = np.load(filepath1)[0]
array1 = np.load(filepath1)
array2 = np.load(filepath2)
array1unpad = array1[206:(len(array1)-206)] # remove pads from both arrays
array2unpad = array2[206:(len(array2)-206)]

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
print(sum(diff2))
print(diff2)

indices = np.arange(0,len(array1))
plt.plot(indices,array1)
plt.plot(indices,array2)
plt.show()
