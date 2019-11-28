from Bio import SeqIO
record = SeqIO.read("chr21.fasta", "fasta")
seq = record.seq
if "N" in seq:
	print ("YES")

#playing around with sequence slicing methods - it is possible we could speed up predictions by splitting string predictions into lists of start/end indexes (?)
#then give a process a index, have it generate a copy of the trained model, and give it a string and the indices to operate on???
#multiprocessing in python: https://docs.python.org/2/library/multiprocessing.html 

def find_polya_peaks_memoryFriendly( seq, sequence_stride=10) :
    #returns peaks found with scipy.signal find_peaks, and the average of the softmax predicted probability for that position in the sequence for every time it is predicted as the window slides along the sequence (this is why the total probability for the sequence being predicted does not add to 1)
	sumCutPreds = np.zeros(len(seq))
	sumMask = np.zeros(len(seq))    
    #set up stat/end position values for slicing sequence string
	start_pos = 0
	end_pos = 205
	while True :
		seq_slice = ''
		effective_len = 0
		if end_pos <= len(seq) : #if the sequence is longer than 205 nts can slice w/o padding
			seq_slice = seq[start_pos: end_pos]
			effective_len = 205
		else : #if sequence is not longer than 205 nts cannot slice w/o padding
			seq_slice = (seq[start_pos:] + ('X' * 200))[:205]
			effective_len = len(seq[start_pos:]) 
		sumCutPreds = sumCutPreds + padded_slice.reshape(1, -1)[:,:-1]
		sumMask = sumMask + padded_mask.reshape(1, -1)[:,:-1]  
		if end_pos >= len(seq) : #if done slicing the seq, we're finished. Break the while
			break
        #update cut positions to predict for next slice 
		start_pos += sequence_stride 
		end_pos += sequence_stride
	avg_cut_pred = sumCutPreds/sumMask
	peak_ixs, _ = find_peaks(avg_cut_pred[0], height=peak_min_height, distance=peak_min_distance, prominence=peak_prominence) 
	return peak_ixs.tolist(), avg_cut_pred
