import numpy as np 

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
			self._previousWeightDelta.append(np.zeros((l2,l1-1)))

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
	bpn = BackPropagationNetwork((26,50,50,5))
	print bpn.weights;

	# lvInput = np.array([[0,0],[1,1],[0,1],[1,0]])
	# lvTarget = np.array([[0.05],[0.05],[0.95],[0.95]])

	# lnMax = 100000
	# lnErr = 1e-5

	# for i in range(lnMax-1):
	# 	err = bpn.train(lvInput,lvTarget,momentum = 0.7)
	# 	if i % 2500 == 0:
	# 		print "Iteration {0} \tError: {1:0.6f}".format(i,err)
	# 	if err <= lnErr:
	# 		print("Minimum error reached at iteration {0}".format(i))
	# 		break


	# lvOutput = bpn.forwardProc(lvInput)
	# print("Output {0}".format(lvOutput))

	