import keras
from keras.models import Sequential, Model, load_model
from keras import backend as K

import tensorflow as tf

import os
import time
import numpy as np

from scipy.signal import convolve as sp_conv
from scipy.signal import correlate as sp_corr
from scipy.signal import find_peaks


#performs as expected with seq_lenght long strings
#if strings fed to encoder are longer than seq_length, encodes up to the end then encodes all zero rows for 'missing' letters
#if strings are shorter than the seq_length will crash
class OneHotEncoder :
    def __init__(self, seq_length=100, default_fill_value=0) :
        #creates class capable of encoding a seq_length long amount of nucleotides
        self.seq_length = seq_length
        self.default_fill_value = default_fill_value #is 0 unless otherwise specified
        self.encode_map = {
            'A' : 0,
            'C' : 1,
            'G' : 2,
            'T' : 3
        }
        self.decode_map = {
            0 : 'A',
            1 : 'C',
            2 : 'G',
            3 : 'T',
            -1 : 'X'
        }
    
    def encode(self, seq) :
        #encode string into self rows # 4 cols array
        one_hot = np.zeros((self.seq_length, 4)) 
        self.encode_inplace(seq, one_hot)
        return one_hot
    
    def encode_inplace(self, seq, encoding) :
        #for each position, fill with corresponding value in the encode_map or with teh default fill value (0 by default)
        for pos, nt in enumerate(list(seq)) :
            if nt in self.encode_map :
                encoding[pos, self.encode_map[nt]] = 1
            elif self.default_fill_value != 0 :
                encoding[pos, :] = self.default_fill_value
    
    def __call__(self, seq) :
        return self.encode(seq)

def logit(x) :
    #https://www.theanalysisfactor.com/what-is-logit-function/ logit function intro
    #TODO: FIGURE OUT WHAT WE ARE USING THIS FOR
	return np.log(x / (1.0 - x))

def get_aparent_encoder(lib_bias=None) :
    #set up one hot encoder for them model
    #TODO: WHAT IS LIB BIAS
	onehot_encoder = OneHotEncoder(205) #autoencoder can encode 205, 205 + long sequences
    
    #reshapes the 205x4 onehot of sequence into 
	def encode_for_aparent(sequences) :
		one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1)) for sequence in sequences], axis=0)
		
		fake_lib = np.zeros((len(sequences), 13)) #which of the 13 libraries it is from
		fake_d = np.ones((len(sequences), 1))

		if lib_bias is not None : #if there is a lib_bias, sets that row to 1 in the fake_lib 
			fake_lib[:, lib_bias] = 1.

		return [ 
			one_hots,
			fake_lib,
			fake_d
		]
    
	return encode_for_aparent

#PROB DON"T NEED####################
def get_aparent_legacy_encoder(lib_bias=None) :
	onehot_encoder = OneHotEncoder(185)

	def encode_for_aparent(sequences) :
		one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, 1, len(sequence), 4)) for sequence in sequences], axis=0)
		
		fake_lib = np.zeros((len(sequences), 36))
		fake_d = np.ones((len(sequences), 1))
		
		if lib_bias is not None :
			fake_lib[:, lib_bias] = 1.

		return [
			one_hots,
			fake_lib,
			fake_d
		]

	return encode_for_aparent
###################################

#MAYBE WILL NEED
def get_apadb_encoder() :
	onehot_encoder = OneHotEncoder(205)

	def encode_for_apadb(prox_sequences, dist_sequences, prox_cut_starts, prox_cut_ends, dist_cut_starts, dist_cut_ends, site_distances) :
        #encode proximal and distal sequencesfrom apadb
		prox_one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1)) for sequence in prox_sequences], axis=0)
		dist_one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1)) for sequence in dist_sequences], axis=0)
        
        #The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions. -> from Numpy.reshape docs
		return [
			prox_one_hots,
			dist_one_hots,
			np.array(prox_cut_starts).reshape(-1, 1),
			np.array(prox_cut_ends).reshape(-1, 1),
			np.array(dist_cut_starts).reshape(-1, 1),
			np.array(dist_cut_ends).reshape(-1, 1),
			np.log(np.array(site_distances).reshape(-1, 1)), #takes log of site_distances stored in apadb
			np.zeros((len(prox_sequences), 13)),
			np.ones((len(prox_sequences), 1))
		]
    
	return encode_for_apadb

def find_polya_peaks(aparent_model, aparent_encoder, seq, sequence_stride=10, conv_smoothing=True, peak_min_height=0.01, peak_min_distance=50, peak_prominence=(0.01, None)) :
    #returns peaks found with scipy.signal find_peaks, and the average of the softmax predicted probability for that position in the sequence for every time it is predicted as the window slides along the sequence (this is why the total probability for the sequence being predicted does not add to 1)
	cut_pred_padded_slices = []
	cut_pred_padded_masks = []
    
    #set up stat/end position values for slicing sequence string
	start_pos = 0
	end_pos = 205
	print ("BEGIN")
	while True :

		seq_slice = ''
		effective_len = 0

		if end_pos <= len(seq) : #if the sequence is longer than 205 nts can slice w/o padding
			seq_slice = seq[start_pos: end_pos]
			effective_len = 205
		else : #if sequence is not longer than 205 nts cannot slice w/o padding
			seq_slice = (seq[start_pos:] + ('X' * 200))[:205] #pad with 'X', will be filled with 0 in one hot encoder (unless set to other value)
			effective_len = len(seq[start_pos:]) #effective length is only the length of the actual string, not string + padding

		_, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice])) #predicts for the sequence slice just constructed
		#print ("sum of cut_pred one pred: ", sum(cut_pred[0]))
		#print (len(cut_pred[0]))        
		#print("Striding over subsequence [" + str(start_pos) + ", " + str(end_pos) + "] (Total length = " + str(len(seq)) + ")...")
		#print ("seq_slice")
		#print (seq_slice)        
		#print ("cut slice")
		#print (cut_pred)
		padded_slice = np.concatenate([
			np.zeros(start_pos), #0's before sequence slice
			np.ravel(cut_pred)[:effective_len], #sequence slice predictions
			np.zeros(len(seq) - start_pos - effective_len), #zeros after predictions
			np.array([np.ravel(cut_pred)[205]]) #TODO What does this do???
		], axis=0)
		#print ("padded_slice: ")
		#print (padded_slice)        
		padded_mask = np.concatenate([
			np.zeros(start_pos),
			np.ones(effective_len),
			np.zeros(len(seq) - start_pos - effective_len),
			np.ones(1)
		], axis=0)[:len(seq)+1]
		#print ("padded_mask: ")
		#print (padded_mask)
		cut_pred_padded_slices.append(padded_slice.reshape(1, -1))
		cut_pred_padded_masks.append(padded_mask.reshape(1, -1))
		#print ("cut_pred_padded_slices: ")
		#print (cut_pred_padded_slices)
		#print ("cut_pred_padded_slices: ")
		#print (cut_pred_padded_masks)        

		if end_pos >= len(seq) : #if done slicing the seq, we're finished. Break the while
			break
        
        #update cut positions to predict for next slice 
		start_pos += sequence_stride 
		end_pos += sequence_stride

    
	cut_slices = np.concatenate(cut_pred_padded_slices, axis=0)[:, :-1] #concatenate all padded slice stuff
	cut_masks = np.concatenate(cut_pred_padded_masks, axis=0)[:, :-1]
	#print ("cut_slices: ")
	#print (cut_slices)
	#print (cut_slices.shape)
	#print ("cut_masks: ")
	#print (cut_masks)
	#print (cut_masks.shape)
	#print (sum(cut_masks))    
    #scipy.signal.correlate smoothing 
	if conv_smoothing :
		smooth_filter = np.array([
			[0.005, 0.01, 0.025, 0.05, 0.085, 0.175, 0.3, 0.175, 0.085, 0.05, 0.025, 0.01, 0.005]
		])

		cut_slices = sp_corr(cut_slices, smooth_filter, mode='same') 
	
	#divide the sum of the cut_slices by the masks
	avg_cut_pred = np.sum(cut_slices, axis=0) / np.sum(cut_masks, axis=0)
	#print("avg_cut_pred per site: ")
	#print(avg_cut_pred)    
	std_cut_pred = np.sqrt(np.sum((cut_slices - np.expand_dims(avg_cut_pred, axis=0))**2, axis=0) / np.sum(cut_masks, axis=0))
	#print("std_cut_pred per site: ")
	#print(std_cut_pred)
	peak_ixs, _ = find_peaks(avg_cut_pred, height=peak_min_height, distance=peak_min_distance, prominence=peak_prominence) #using scipy signal function find_peaks 

	
	return peak_ixs.tolist(), avg_cut_pred, sum(cut_masks)

def find_polya_peaks_tofile(fName, aparent_model, aparent_encoder, seq, sequence_stride=10, conv_smoothing=True, peak_min_height=0.01, peak_min_distance=50, peak_prominence=(0.01, None) ) :
    #returns peaks found with scipy.signal find_peaks, and the average of the softmax predicted probability for that position in the sequence for every time it is predicted as the window slides along the sequence (this is why the total probability for the sequence being predicted does not add to 1)
    #since the memory requirement/whatever keeps killing the jupyter kernel is really annoying and likely has to do with memory limits, will be cutting the sequence into smaller cut prediction windows and storing the numpy arrays as binaries for hte pred and mask, then will assemble those in another function to see if it helps with the repeated death issue
	fileOut = 0
    #set up stat/end position values for slicing sequence string
	start_pos = 0
	end_pos = 205
	print ("BEGIN outputting: ")
	while True :

		seq_slice = ''
		effective_len = 0
		if end_pos <= len(seq) : #if the sequence is longer than 205 nts can slice w/o padding
			seq_slice = seq[start_pos: end_pos]
			effective_len = 205
		else : #if sequence is not longer than 205 nts cannot slice w/o padding
			seq_slice = (seq[start_pos:] + ('X' * 200))[:205] #pad with 'X', will be filled with 0 in one hot encoder (unless set to other value)
			effective_len = len(seq[start_pos:]) #effective length is only the length of the actual string, not string + padding

		_, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice])) #predicts for the sequence slice just constructed
		padded_slice = np.concatenate([
			np.zeros(start_pos), #0's before sequence slice
			np.ravel(cut_pred)[:effective_len], #sequence slice predictions
			np.zeros(len(seq) - start_pos - effective_len), #zeros after predictions
			np.array([np.ravel(cut_pred)[205]]) #TODO What does this do???
		], axis=0)       
		padded_mask = np.concatenate([
			np.zeros(start_pos),
			np.ones(effective_len),
			np.zeros(len(seq) - start_pos - effective_len),
			np.ones(1)
		], axis=0)[:len(seq)+1]
		
		#outputting npy for the slices and the masks
		np.save(fName + "PredSlices" + str(fileOut), padded_slice.reshape(1, -1))
		np.save(fName + "MaskSlices" + str(fileOut), padded_mask.reshape(1, -1))
		if end_pos >= len(seq) : #if done slicing the seq, we're finished. Break the while
			break

        #update cut positions to predict for next slice 
		start_pos += sequence_stride 
		end_pos += sequence_stride
		fileOut += 1

	return fileOut



def find_polya_peaks_memoryFriendly(aparent_model, aparent_encoder, seq, sequence_stride=10, conv_smoothing=True, peak_min_height=0.01, peak_min_distance=50, peak_prominence=(0.01, None)) :
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
		#	print (len(seq_slice))
			effective_len = 205
		else : #if sequence is not longer than 205 nts cannot slice w/o padding
			seq_slice = (seq[start_pos:] + ('X' * 200))[:205]
			effective_len = len(seq[start_pos:]) 
		_, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice])) #predicts for the sequence slice just constructed
		#print (cut_pred.shape)
		#print ("up to effective len: ")
		#print (np.ravel(cut_pred)[:effective_len])
		#print (np.ravel(cut_pred)[:effective_len].shape)
		#print ("last pos???")
		#print (np.array([np.ravel(cut_pred)[205]]))
		#print (np.array([np.ravel(cut_pred)[205]]).shape)
		#print ( "--------------------")
		padded_slice = np.concatenate([
			np.zeros(start_pos), #0's before sequence slice
			np.ravel(cut_pred)[:effective_len], #sequence slice predictions
			np.zeros(len(seq) - start_pos - effective_len), #zeros after predictions
			np.array([np.ravel(cut_pred)[205]]) #TODO What does this do???
		], axis=0)       
		padded_mask = np.concatenate([
			np.zeros(start_pos),
			np.ones(effective_len),
		np.zeros(len(seq) - start_pos - effective_len),
			np.ones(1)
		], axis=0)[:len(seq)+1]
		#print ("--------------------------------")
		#print ("prereshape:")
		#print (padded_slice.shape)
		#print (padded_mask.shape)
		reshapedSlice = padded_slice.reshape(1, -1)[:,:-1]
		reshapedMask = padded_mask.reshape(1, -1)[:,:-1]  
		#print ("reshape:")
		#print (reshapedSlice.shape)
		#print (reshapedMask.shape)
		#print ("---------------------")
		sumCutPreds = sumCutPreds + reshapedSlice
		#print (" sum shape: ", sumCutPreds.shape)
		sumMask = sumMask + reshapedMask
		if end_pos >= len(seq) : #if done slicing the seq, we're finished. Break the while
			break
        #update cut positions to predict for next slice 
		start_pos += sequence_stride 
		end_pos += sequence_stride
	avg_cut_pred = sumCutPreds/sumMask
	#print ("avg shape: ", avg_cut_pred.shape)
	#print ("avg inside: ", avg_cut_pred[0].shape)
	peak_ixs, _ = find_peaks(avg_cut_pred[0], height=peak_min_height, distance=peak_min_distance, prominence=peak_prominence) 
	return peak_ixs.tolist(), avg_cut_pred

#no padding version (reduces number operations)
def find_polya_peaks_memoryFriendlyV2(aparent_model, aparent_encoder, seq, sequence_stride=10, conv_smoothing=True, peak_min_height=0.01, peak_min_distance=50, peak_prominence=(0.01, None)) :
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
			#print (len(seq_slice))
			effective_len = 205
		else : #if sequence is not longer than 205 nts cannot slice w/o padding
			seq_slice = (seq[start_pos:] + ('X' * 200))[:205]
			effective_len = len(seq[start_pos:]) 
		#print (effective_len)
		_, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice])) #predicts for the sequence slice just constructed
		pred_removeLast = np.ravel(cut_pred)[:effective_len]      
		#print ( pred_removeLast.shape)
		mask_removeLast = np.ones(effective_len)
		#print (sumCutPreds[ effective_len].shape)
		#print (pred_removeLast.shape)
		sumCutPreds[start_pos : start_pos + effective_len] = sumCutPreds[start_pos : start_pos + effective_len] + pred_removeLast
		sumMask[start_pos : start_pos + effective_len] = sumMask[start_pos : start_pos + effective_len] + mask_removeLast
		if end_pos >= len(seq) : #if done slicing the seq, we're finished. Break the while
			break 
		start_pos += sequence_stride 
		end_pos += sequence_stride
	avg_cut_pred = sumCutPreds/sumMask
	peak_ixs, _ = find_peaks(avg_cut_pred, height=peak_min_height, distance=peak_min_distance, prominence=peak_prominence) 
	return peak_ixs.tolist(), avg_cut_pred, sumMask


#no padding version (reduces number operations)
def find_polya_peaks_memoryFriendlyV3(aparent_model, aparent_encoder, seq, fileStem, sequence_stride, output_size) :
    #returns peaks found with scipy.signal find_peaks, and the average of the softmax predicted probability for that position in the sequence for every time it is predicted as the window slides along the sequence (this is why the total probability for the sequence being predicted does not add to 1)
	sumCutPreds = np.zeros(205)
	sumMask = np.zeros(205)    
	fileNames = [] #list of numpy binary files created 
    #set up stat/end position values for slicing sequence string
	start_pos = 0
	end_pos = 205
	files_out = 1
	totalTimeStart = time.time()
	while True :
		start_time = time.time()
		seq_slice = ''
		effective_len = 0
		if end_pos <= len(seq): #if the sequence is longer than 205 nts can slice w/o padding
			seq_slice = seq[start_pos: end_pos]
			#print (len(seq_slice))
			effective_len = 205
		else : #if sequence is not longer than 205 nts cannot slice w/o padding
			seq_slice = (seq[start_pos:] + ('X' * 200))[:205]
			effective_len = len(seq[start_pos:]) 
		_, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice])) #predicts for the sequence slice just constructed
		pred_removeLast = np.ravel(cut_pred)[:effective_len]      
		mask_removeLast = np.ones(effective_len)
		#print (sumCutPreds[- effective_len:].shape)
		#print (pred_removeLast.shape)
		sumCutPreds[- effective_len:] = sumCutPreds[- effective_len:] + pred_removeLast
		#print (sumMask)
		#print (sumMask[-effective_len:])
		#print ( mask_removeLast.shape)
		sumMask[-effective_len:] = sumMask[-effective_len:] + mask_removeLast
		#print (sumMask[-effective_len:])
		#print ("buffer zone")
		
		if end_pos >= len(seq) : #if done slicing the seq, we're finished. Break the while and export 
			
			#output final length of predictions	
			outAvg = sumCutPreds/sumMask
			print ("final shape: ", outAvg.shape)
			print ("END POS: ", end_pos, " len(seq): ", len(seq))
			name_current = fileStem + "_" + str(files_out) + "_" + str(output_size)
			fileNames.append(name_current)
			np.save(name_current, outAvg)
			#print ("On final file: ", files_out, "time: ", time.time()-start_time)
			
			break 
		start_pos += sequence_stride 
		end_pos += sequence_stride
		#increase length by stride for sumCutPreds and sumMask
		sumCutPreds = np.concatenate((sumCutPreds, np.zeros(sequence_stride)), axis = 0)
		sumMask = np.concatenate((sumMask, np.zeros(sequence_stride)), axis = 0)
		#see if there is enough done to export (buffer zone of 500 long)
		
		if sumCutPreds.size >= output_size + 500:
			#export the first output_size of the working edge averaged, reset the sumCutPreds and sumMask
			predOut = sumCutPreds[:output_size]
			maskOut = sumMask[:output_size]
			sumCutPreds = sumCutPreds[output_size:]
			sumMask = sumMask[output_size:]
			outAvg = predOut/maskOut
			name_current = fileStem + "_" + str(files_out) + "_" + str(output_size)
			fileNames.append(name_current)
			np.save(name_current, outAvg)
			#print ("On file: ", files_out, "time: ", time.time()-start_time)
			files_out += 1
		
	#avg_cut_pred = sumCutPreds/sumMask
	#peak_ixs, _ = find_peaks(avg_cut_pred, height=peak_min_height, distance=peak_min_distance, prominence=peak_prominence) 
	print ("total elapsed time: ", time.time() - totalTimeStart)
	#avgOut = sumCutPreds/sumMask
	return fileNames #, avgOut, sumMask



##########################################
def score_polya_peaks(aparent_model, aparent_encoder, seq, peak_ixs, sequence_stride=2, strided_agg_mode='max', iso_scoring_mode='both', score_unit='log') :
	peak_iso_scores = []

	iso_pred_dict = {}
	iso_pred_from_cuts_dict = {}

	for peak_ix in peak_ixs :

		iso_pred_dict[peak_ix] = []
		iso_pred_from_cuts_dict[peak_ix] = []

		if peak_ix > 75 and peak_ix < len(seq) - 150 :
			for j in range(0, 30, sequence_stride) :
				seq_slice = (('X' * 35) + seq + ('X' * 35))[peak_ix + 35 - 80 - j: peak_ix + 35 - 80 - j + 205]

				if len(seq_slice) != 205 :
					continue

				iso_pred, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice]))

				iso_pred_dict[peak_ix].append(iso_pred[0, 0])
				iso_pred_from_cuts_dict[peak_ix].append(np.sum(cut_pred[0, 77: 107]))

		if len(iso_pred_dict[peak_ix]) > 0 :
			iso_pred = np.mean(iso_pred_dict[peak_ix])
			iso_pred_from_cuts = np.mean(iso_pred_from_cuts_dict[peak_ix])
			if strided_agg_mode == 'max' :
				iso_pred = np.max(iso_pred_dict[peak_ix])
				iso_pred_from_cuts = np.max(iso_pred_from_cuts_dict[peak_ix])
			elif strided_agg_mode == 'median' :
				iso_pred = np.median(iso_pred_dict[peak_ix])
				iso_pred_from_cuts = np.median(iso_pred_from_cuts_dict[peak_ix])

			if iso_scoring_mode == 'both' :
				peak_iso_scores.append((iso_pred + iso_pred_from_cuts) / 2.)
			elif iso_scoring_mode == 'from_iso' :
				peak_iso_scores.append(iso_pred)
			elif iso_scoring_mode == 'from_cuts' :
				peak_iso_scores.append(iso_pred_from_cuts)

			if score_unit == 'log' :
				peak_iso_scores[-1] = np.log(peak_iso_scores[-1] / (1. - peak_iso_scores[-1]))


			peak_iso_scores[-1] = round(peak_iso_scores[-1], 3)
		else :
			peak_iso_scores.append(-10)
	
	return peak_iso_scores

