import numpy as np 
import time

class BackPropagationNetwork:

	layerCount = 0;
	shape  = None;
	weights = [];

	def __init__(self,layerSize):

		self.layerCount = len(layerSize) - 1;
		self.shape = layerSize

		self._layerInput = []
		self._layerOutput = []
		self._previousWeightDelta = []

		for (l1,l2) in zip(layerSize[:-1],layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))
			self._previousWeightDelta.append(np.zeros((l2,l1+1)))

	def forwardProc(self,input):

		InCases = input.shape[0]

		self._layerInput = []
		self._layerOutput = []

		for index in range(self.layerCount):
			if index == 0:
				#print "weight" + str(self.weights[0])
				#print "vstack" + str(np.vstack([input.T,np.ones([1,InCases])]))
				layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,InCases])]))
				#print "layerInput" + str(layerInput)
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,InCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))

		return self._layerOutput[-1].T

	def train(self,input,target, trainingRate = 0.2, momentum = 0.5):

		delta = []
		InCases = input.shape[0]

		self.forwardProc(input)

		#Delta calculation
		for index in reversed(range(self.layerCount)):

			if index == self.layerCount - 1 :

				output_delta = self._layerOutput[index] - target.T
				error = np.sum(output_delta**2)
				delta.append(output_delta * self.sgm(self._layerInput[index],True))

			else:

				delta_pullback = self.weights[index+1].T.dot(delta[-1])
				delta.append(delta_pullback[:-1,:] * self.sgm(self._layerInput[index],True))

		#Weight Delta Calculation
		for index in range(self.layerCount):
			delta_index  = self.layerCount - 1 - index

			if index == 0:
				layerOutput  = np.vstack([input.T,np.ones([1,InCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[index-1],np.ones([1,self._layerOutput[index-1].shape[1]])])

			currWeightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0),axis = 0)

			weightDelta = trainingRate * currWeightDelta + momentum * self._previousWeightDelta[index]

			self.weights[index] -= weightDelta

			self._previousWeightDelta[index] = weightDelta

		return error

	def sgm(self,x,Derivative=False):
		if not Derivative:
			return 1/ (1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)


if __name__ == "__main__":
	bpn = BackPropagationNetwork((260,50,5))
	

	f1 = open("mfccData/A_mfcc.npy")
	f2 = open("mfccData/E_mfcc.npy")
	f3 = open("mfccData/I_mfcc.npy")
	f4 = open("mfccData/O_mfcc.npy")
	f5 = open("mfccData/U_mfcc.npy")
	
	inputArray1  = np.load(f1)
	inputArray2  = np.load(f2)
	inputArray3  = np.load(f3)
	inputArray4  = np.load(f4)
	inputArray5  = np.load(f5)
	inputArray = np.concatenate((inputArray1,inputArray2,inputArray3,inputArray4,inputArray5))

	print inputArray.shape

	t1 = np.array([[1,0,0,0,0] for _ in range(len(inputArray1))])
	t2 = np.array([[0,1,0,0,0] for _ in range(len(inputArray2))])
	t3 = np.array([[0,0,1,0,0] for _ in range(len(inputArray3))])
	t4 = np.array([[0,0,0,1,0] for _ in range(len(inputArray4))])
	t5 = np.array([[0,0,0,0,1] for _ in range(len(inputArray5))])

	target = np.concatenate([t1,t2,t3,t4,t5])
	print target.shape	

	lnMax = 1000000
	lnErr = 1e-5

	startTime = time.clock()

	#Train Loop
	for i in range(lnMax-1):
		err = bpn.train(inputArray,target,momentum = 0.3)
		if i % 1500 == 0:
			print "Iteration {0} \tError: {1:0.6f}".format(i,err)
		if err <= lnErr:
			print("Minimum error reached at iteration {0}".format(i))
			break

	endTime = time.clock()

	with open("network/" + "vowel_network_2"+ ".npy", 'w') as outfile:
  		np.save(outfile,bpn.weights)

  	

	lvOutput = bpn.forwardProc(inputArray)
	print("Output {0}".format(lvOutput))

	print "Time Elapsed: " + str(endTime - startTime) + " seconds"
	print "Total Iteration {0} \t Total Error: {1:0.6f}".format(i,err)
	