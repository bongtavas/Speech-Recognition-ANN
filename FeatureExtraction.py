from features import mfcc
from operator import add
import scipy.io.wavfile as wav
import numpy as np

vowels = ['A','E','I','O','U']

for x in range(len(vowels)):
	fileString = vowels[x]+"_mfcc"
	data = []
	for i in range(10):
		(rate,sig) = wav.read("sound_files/"+ vowels[x] + "-" + str(i+1) + ".wav")
		print "Reading: " + vowels[x] + "-" + str(i+1) + ".wav"
		mfcc_feat = mfcc(sig,rate)

		s = mfcc_feat[:20]
		st = []
		for elem in s:
			st.extend(elem)
		
		st /= np.max(np.abs(st),axis=0)
		data.append(st)
		print st
		
	with open("mfccData/" + fileString+ ".npy", 'w') as outfile:
   		np.save(outfile,data)




