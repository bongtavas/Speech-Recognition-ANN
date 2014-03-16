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

	f1 = open("network/vowel_network.npy")
	f2 = open("mfccData/U_mfcc.npy")
	
	weights  = np.load(f1)
	inputArray = np.load(f2)

	testNet = TestingNetwork((260,16,5),weights)

	lvOutput = testNet.forwardProc(inputArray)
	print("Output {0}".format(lvOutput))
