import numpy as np 
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

	
	weights  = np.load(f1)
	
	inputArray = []
	for i in range(10)[6:10]:
		(rate,sig) = wav.read("sound_files/"+ vowels[x] + "-" + str(i+1) + ".wav")
		print "Reading: " + vowels[x] + "-" + str(i+1) + ".wav"
		mfcc_feat = mfcc(sig,rate)

		s = mfcc_feat[:20]
		st = []
		for elem in s:
			st.extend(elem)
		
		st /= np.max(np.abs(st),axis=0)
		inputArray.append(st)
		

	testNet = TestingNetwork((260,16,5),weights)

	lvOutput = testNet.forwardProc(inputArray)
	print("Output {0}".format(lvOutput))
