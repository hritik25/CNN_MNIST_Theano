import random
import numpy as np

class CrossEntropy(object):

	@staticmethod
	def funtion(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@staticmethod
	def delta(a, y):
		return (a-y)

class QuadraticCost(object):
	
	@staticmethod
	def function(a, y):
		return 0.5*np.sum(np.sqaure(a-y))

	@staticmethod
	def delta(a, y):
		return (a-y)*sigmoidPrime(a)


class Network(object):

	def __init__(self, sizes, cost=CrossEntropy):
		# sizes is a list containing no. of units in each layer
		self.numLayers = len(sizes)
		self.sizes = sizes
		self.defaultWeightInitializer()
		self.cost = cost

	def largeWeightInitializer(self):
		# list of (num_layers-1) weight matrices,so if their is one hidden layer in the network,
		# making it a total of 3 layers, we will have two weight matrices.
		self.weights = [ np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:]) ]
		self.biases = [ np.random.randn(y,1) for y in self.sizes[1:] ]

	def defaultWeightInitializer(self):
		# intializing weights as random Gaussian variables with mean 0, 
		# and std. deviation 1/sqrt(no. of input neurons)
		self.weights = [ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:]) ]
		self.biases = [ np.random.randn(y,1) for y in self.sizes[1:] ]

	def feedForward(self, x):
		activation = x
		for w,b in zip(self.weights, self.biases):
			activation = sigmoid(np.dot(w, activation) + b)
		return activation

	def stGradientDescent(self, trd, epochs, batchSize, learningRate, lmbda, testData=None):
		"""
		for given no. of epochs :
		1. randomly shuffle the training data
		2. divide the training data into mini batches 
		3. call updataMiniBatch on every batch 
		"""
		for j in xrange(0, epochs):
			random.shuffle(trd)
			miniBatches = [ trd[k:k+batchSize] for k in xrange(0,len(trd),batchSize) ]
			for miniBatch in miniBatches:
				self.updateMiniBatch( miniBatch, learningRate, lmbda, len(trd) )
			if testData:
				score = self.evaluate(testData)
				print "Epoch {0} complete. Accuracy = {1}".format(j, float(score)/100)

	def updateMiniBatch(self, miniBatch, learningRate, lmbda, n):
		"""
		1.find deltaW and deltaB, i.e. the improvement in weights
		  and biases by backpropagation of error calculated in a 
		  training sample
		2.update the weights and biases
		"""
		sumDeltaW = [ np.zeros(w.shape) for w in self.weights ]
		sumDeltaB = [ np.zeros(b.shape) for b in self.biases ]
		# for w,b in zip(sumDeltaW, sumDeltaB):
		# 	print w.shape, b.shape
		for (x,y) in miniBatch:
			deltaW, deltaB = self.backProp(x,y)
			sumDeltaW = [ sdw + dw for sdw, dw in zip(sumDeltaW, deltaW) ]
			sumDeltaB = [ sdb + db for sdb, db in zip(sumDeltaB, deltaB) ]
		self.weights = [(1-learningRate*lmbda/n)*w - (learningRate/len(miniBatch))*dw for w,dw in zip(self.weights, sumDeltaW) ]
		self.biases = [ b - (learningRate/len(miniBatch))*db for b,db in zip(self.biases, sumDeltaB) ]

	def backProp(self, x, y):
		deltaW = [ np.zeros(w.shape) for w in self.weights ]
		deltaB = [ np.zeros(b.shape) for b in self.biases ]
		activations = []
		activation = x
		activations.append(activation)
		for w,b in zip(self.weights, self.biases):
			z = np.dot(w,activation)+b
			activation = sigmoid(z)
			activations.append(activation)
		delta = (self.cost).delta(activations[-1], y)
		deltaW[-1] = np.dot(delta, activations[-2].T)
		deltaB[-1] = delta
		for l in range(2, self.numLayers):
			delta = np.dot(self.weights[-l+1].T,delta)*sigmoidPrime(activations[-l])
			deltaW[-l] = np.dot(delta, activations[-l-1].T)
			deltaB[-l] = delta
		return (deltaW, deltaB)

	def evaluate(self, testData):
		score = 0
		for x,y in testData:
			output = self.feedForward(x)
			score += int(np.argmax(output)==y)
		return score

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def sigmoidPrime(a):
    """Derivative of the sigmoid function."""
    return a*(1-a)