import random
import numpy as np

class Network(object):

	def __init__(self, sizes):
		# sizes is a list containing no. of units in each layer
		self.layers = len(sizes) 
		# weights : if their is one hidden layer in the network,
		# making it a total of 3 layers, we will have two weight matrices
		self.weights = [ np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:]) ]
		self.biases = [ np.random.randn(y,1) for y in sizes[1:] ]
		print self.weights

	def feedForward(self, x):
		activation = x
		for w,b in zip(self.weights, self.biases):
			activation = sigmoid(np.dot(w, activation) + b)
		return activation

	def stGradientDescent(self, trd, epochs, batchSize, learningRate, testData=None):
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
				self.updateMiniBatch( miniBatch, learningRate )
			if testData:
				score = self.evaluate(testData)
				print "Epoch {0} complete. Score = {1}".format(j, score)

	def updateMiniBatch(self, miniBatch, learningRate):
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
		self.biases = [b - (learningRate/len(miniBatch))*db for b,db in zip(self.biases, sumDeltaB) ]

	def backProp(self, x, y):
		deltaW = [ np.zeros(w.shape) for w in self.weights ]
		deltaB = [ np.zeros(b.shape) for b in self.biases ]
		activations = []
		zs = []
		activation = x
		activations.append(activation)
		for w,b in zip(self.weights, self.biases):
			z = np.dot(w,activation)+b
			activation = sigmoid(z)
			activations.append(activation)
		delta = costDerivative(activations[-1], y)*sigmoidPrime(activations[-1])
		deltaW[-1] = np.dot(delta, activations[-2].T)
		deltaB[-1] = delta
		for l in range(2, self.layers):
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

def costDerivative(outputActivation, y):
	return (outputActivation-y)

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def sigmoidPrime(a):
    """Derivative of the sigmoid function."""
    return a*(1-a)