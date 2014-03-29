from __future__ import division
from features import mfcc
from operator import add
import scipy.io.wavfile as wav
import numpy as np

words = ['apple','banana','kiwi','lime','orange']


for x in range(len(words)):
	fileString = words[x]+"_mfcc"
	data = []
	for i in range(10):
		(rate,sig) = wav.read("training_sets/"+ words[x] + "-" + str(i+1) + ".wav")
		print "Reading: " + words[x] + "-" + str(i+1) + ".wav"
		duration = len(sig)/rate
		mfcc_feat = mfcc(sig,rate,winlen=duration/20,winstep=duration/20)
		s = mfcc_feat[:20]
		st = []
		for elem in s:
			st.extend(elem)
		
		st /= np.max(np.abs(st),axis=0)
		data.append(st)
		print st
		
	with open("mfccData/" + fileString+ ".npy", 'w') as outfile:
   		np.save(outfile,data)




