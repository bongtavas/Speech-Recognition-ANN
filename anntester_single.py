from __future__ import division 
import numpy as np 
import scipy.io.wavfile as wav

from features import mfcc

class TestingNetwork:

	layerCount = 0;
	shape  = None;
	weights = [];

	def __init__(self,layerSize,weights):

		self.layerCount = len(layerSize) - 1;
		self.shape = layerSize

		self._layerInput = []
		self._layerOutput = []
		self.weights = weights
	def forwardProc(self,input):

		InCases = input.shape[0]

		self._layerInput = []
		self._layerOutput = []

		for index in range(self.layerCount):
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,InCases])]))
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,InCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))

		return self._layerOutput[-1].T

	def sgm(self,x,Derivative=False):
		if not Derivative:
			return 1/ (1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)


if __name__ == "__main__":

	f1 = open("network/vowel_network_2.npy")
	
	#Get MFCC Feature Array
	(rate,sig) = wav.read("test_files/test5.wav")
	x = ((3*rate)/(len(sig)-256))
	print x
	mfcc_feat = mfcc(sig,rate)
	
	# print mfcc_feat
	# print len(mfcc_feat)
	# s = mfcc_feat[:20]
	# st = []
	# for elem in s:
	# 	st.extend(elem)
	# st /= np.max(np.abs(st),axis=0)
	# inputArray = np.array([st])

	# #Setup Neural Network
	# weights  = np.load(f1)
	# testNet = TestingNetwork((260,50,5),weights)


	# #Input MFCC Array to Network
	# outputArray = testNet.forwardProc(inputArray)
	# print outputArray;
	# indexMax = outputArray.argmax(axis = 1)[0]

	# if indexMax == 0:
	# 	print "Detected: Vowel A";
	# elif indexMax==1:
	# 	print "Detected: Vowel E";
	# elif indexMax==2:
	# 	print "Detected: Vowel I";
	# elif indexMax==3:
	# 	print "Detected: Vowel O";
	# elif indexMax==4:
	# 	print "Detected: Vowel U";
		


