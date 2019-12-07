# compare merged vs full run for FAsdfsd

import numpy as np
from matplotlib import pyplot as plt

directory = '../PredictionBinaries/FA270747.1Predictions/compare/'
directory1= '../PredictionBinaries/CM000681.2Predictions/compare/'
#directory = '../PredictionBinaries/KK270747.1Predictions/compare/'

# master is filename1
#filename1 = 'FA270747_1_cutPredsStrideLen10SliceSize35209TotLen35210_merged.npy'
filename1 = 'FA270747_1_cutPredsStrideLen50SliceNum0Start0End35209TotLen35210.npy'
filename2 = 'FA270747_1_cutPredsStrideLen50SliceSize1571TotLen35210_merged.npy'
filename3 = 'FA270747_1_cutPredsStrideLen50SliceNum0Start0End1571TotLen35210.npy'
filename4 = 'FA270747_1_cutPredsStrideLen50SliceNum1Start1000End2571TotLen35210.npy'
filename5 = 'FA270747_1_cutPredsStrideLen50SliceNum2Start2000End3571TotLen35210.npy'
filename6 = 'FA270747_1_cutPredsStrideLen50SliceSize1799TotLen35210_merged.npy'
filename7 = 'FA270747_1_cutPredsStrideLen50SliceSize1799TotLen35210_merged2.npy'
filename8 = 'FA270747_1_cutPredsStrideLen50SliceNum0Start0End1799TotLen35210.npy'
filename9 = 'FA270747_1_cutPredsStrideLen50SliceNum1Start1000End2799TotLen35210.npy'
filename10= 'FA270747_1_cutPredsStrideLen50SliceNum27Start27000End28799TotLen35210.npy'
filename11= 'FA270747_1_cutPredsStrideLen50SliceNum27Start27000End28799TotLen35210.npy'
filename12= 'FA270747_1_cutPredsStrideLen50SliceSize1799TotLen35210_merged3.npy'
filename13= 'FA270747_1_cutPredsStrideLen50SliceSize1799TotLen35210_merged4.npy'
filename14= 'FA270747_1_cutPredsStrideLen50SliceSize1799TotLen35210_merged5.npy'
filename15= 'FA270747_1_cutPredsStrideLen50SliceSize1799TotLen35210_merged6.npy'
filename16= 'FA270747_1_cutPredsStrideLen50SliceSize1799TotLen35210_merged7.npy'
filename17= 'FA270747_1_cutPredsStrideLen50SliceSize1799TotLen35210_merged8.npy'
filename18= 'CM000681_2_cutPredsStrideLen50SliceSize100799TotLen58617616_merged.npy'
filename19= 'chr19.npy'
#filename1 = 'KI270747_1_cutPredsStrideLen10SliceSize198734TotLen198735_merged.npy'
#filename2 = 'KK270747_1_cutPredsStrideLen10SliceSize100411TotLen198735_merged.npy'

filepath1 = directory+filename1
filepath2 = directory+filename2
filepath3 = directory+filename3
filepath4 = directory+filename4
filepath5 = directory+filename5
filepath6 = directory+filename6
filepath7 = directory+filename7
filepath8 = directory+filename8
filepath9 = directory+filename9
filepath10 = directory+filename10
filepath11 = directory+filename11
filepath12 = directory+filename12
filepath13 = directory+filename13
filepath14 = directory+filename14
filepath15 = directory+filename15
filepath16 = directory+filename16
filepath17 = directory+filename17
filepath18 = directory1+filename18
filepath19 = directory1+filename19

padSize = 206
padSize = 400
diffSize = 572 # difference between slice size and range in file
diffSize = 800
sliceSize = 1000
region1 = np.arange(0,10)
region2 = np.arange(206,216)
region2 = np.arange(202,212)
region3 = np.arange(0,sliceSize+diffSize) # beginning for single/first slice
region4 = np.arange(0,(sliceSize-padSize)) # first slice minus pad
region5 = np.arange(sliceSize,sliceSize*2+diffSize) # second slice
region6 = np.arange(0,sliceSize*2+diffSize) # first two slices combined (theoretically)
region7 = np.arange(sliceSize*2,sliceSize*3+diffSize) # third slice
region8 = np.arange(sliceSize*26,sliceSize*27+diffSize) # 27th slice
region9 = np.arange(sliceSize*27,sliceSize*28+diffSize) # 28th slice


#array1 = np.load(filepath1)[0][region8] # full saved single run array, indexed by a region
array1 = np.load(filepath1)[0] # full single run array
#array1 = np.load(filepath12)[region3] # full merged file from run with 400 size pads
array1 = np.load(filepath18)
array2 = np.load(filepath19)
#array2 = np.load(filepath6) # full merged file from run with 400 size pads
#array2 = np.load(filepath8)[0]   # test array to validate
#array2 = np.load(filepath6)[region3] # load region 3 because this is a single slice
array1unpad = array1[padSize:(len(array1)-padSize)] # remove pads from both arrays
array2unpad = array2[padSize:(len(array2)-padSize)]
#array1[np.abs(array1)<.001] = 0
#array2[np.abs(array2)<.001] = 0
region_all = np.arange(0,len(array1))
region_unpad=region_all[padSize:(len(array1)-padSize)]

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
#print(diff2[300:400])
#print(diff2[400:500])
print(diff2[500:600])
print(diff2[600:700])
print(diff2[700:800])
print(diff2[800:900])
print(diff2[900:940])
print(diff2[940:970])
print(diff2[970:980])
print(diff2[980:990])
print(diff2[990:1003])
print(diff2[998])
print(diff2[999])
print(diff2[1000])

print('debugging:')
print(len(array2))
print(max(diff2))
'''
print(np.argmax(diff2))
print(min(diff2))
print(np.argmin(diff2))
unpad_nonzero_inds1 = region_unpad[abs(diff2)>0]
print(max(unpad_nonzero_inds1))
unpad_nonzero_inds = unpad_nonzero_inds1[unpad_nonzero_inds1<max(unpad_nonzero_inds1)]
unpad_nonzero_vals = diff2[unpad_nonzero_inds]
print(unpad_nonzero_inds)
print(unpad_nonzero_vals)
'''

'''
for i in range(0,len(array1)):
    if(diff2[i]!=0):
        print(i)
'''

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
