import os
import random
import numpy as np
import sklearn.metrics as sm
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784
NUM_NEURONS = 512


# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
	def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.neuronsPerLayer = neuronsPerLayer
		self.lr = learningRate
		self.w1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
		self.w2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

	# Activation function.
	def __sigmoid(self, x):
		return (1 / (1 + np.exp(-x)))

	# Activation prime function.
	def __sigmoidDerivative(self, x):
		val = self.__sigmoid(x)
		return (val * (1 - val))

	# Batch generator for mini-batches. Not randomized.
	def __batchGenerator(self, (x, y), n):
		for i in range(0, len(x), n):
			yield x[i : i + n], y[i : i + n]

	# Training with backpropagation.
	def train(self, xVals, yVals, epochs = 5, minibatches = True, mbs = 100):

		# Train minibatches, epoch number of times with each having 1 fwd and back pass

		for i in range(0, epochs):
			for (xBatch, yBatch) in self.__batchGenerator((xVals, yVals), mbs):
				# Get first 2 layers
				l1, l2 = self.__forward(xBatch)

				# Backpropogate
				self.__backward(l1, l2, xBatch, yBatch)

		return self


	# Forward pass.
	def __forward(self, xVals):

		# Get layer 1 = input . w1
		layer1 = self.__sigmoid(np.dot(xVals, self.w1))

		# Get layer 2 = l1 . w2
		layer2 = self.__sigmoid(np.dot(layer1, self.w2))

		return layer1, layer2

	# Backward pass.
	def __backward(self, l1, l2, xVals, yVals):

		# layer 2 error = l2 - y
		l2e = np.subtract(l2, yVals)

		# layer 2 delta = l2e * f'(l2)
		l2d = l2e * self.__sigmoidDerivative(l2)

		# layer 1 error = l2d * Transpose(W2)
		l1e = np.dot(l2d, self.w2.T)

		# layer 1 delta = l1e * f'(l1)
		l1d = l1e * self.__sigmoidDerivative(l1)

		# layer 1 adj = (Transpose(x) . l1d) * lr
		l1a = (np.dot(xVals.T, l1d)) * self.lr

		# layer 2 adj = (Transpose(l1) . l2d) * lr
		l2a = (np.dot(l1.T, l2d)) * self.lr

		# Update weights
		self.w1 -= l1a
		self.w2 -= l2a

	# Predict.
	def predict(self, xVals):
		_, layer2 = self.__forward(xVals)

		# Creating the array of predictions with only 0s and 1s
		ans = []
		for entry in layer2:
			pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			pred[np.argmax(entry)] = 1
			ans.append(pred)
		return np.array(ans)


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
	ans = []
	for entry in xTest:
		pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		pred[random.randint(0, 9)] = 1
		ans.append(pred)
	return np.array(ans)


#=========================<Pipeline Functions>==================================

def getRawData():
	mnist = tf.keras.datasets.mnist
	(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
	print("Shape of xTrain dataset: %s." % str(xTrain.shape))
	print("Shape of yTrain dataset: %s." % str(yTrain.shape))
	print("Shape of xTest dataset: %s." % str(xTest.shape))
	print("Shape of yTest dataset: %s." % str(yTest.shape))
	return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
	((xTrain, yTrain), (xTest, yTest)) = raw

	# Range reduction here (0-255 ==> 0.0-1.0).
	xTrain, xTest = xTrain/255.0, xTest/255.0

	# Flatten the nd arrays to 1d
	xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2])
	xTest = xTest.reshape(xTest.shape[0], xTest.shape[1] * xTest.shape[2])

	yTrainP = to_categorical(yTrain, NUM_CLASSES)
	yTestP = to_categorical(yTest, NUM_CLASSES)
	print("New shape of xTrain dataset: %s." % str(xTrain.shape))
	print("New shape of xTest dataset: %s." % str(xTest.shape))
	print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
	print("New shape of yTest dataset: %s." % str(yTestP.shape))
	return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
	xTrain, yTrain = data
	if ALGORITHM == "guesser":
		return None   # Guesser has no model, as it is just guessing.

	elif ALGORITHM == "custom_net":
		print("Building and training Custom_NN.")
		neural_net = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, NUM_NEURONS)
		return neural_net.train(xTrain, yTrain)

	elif ALGORITHM == "tf_net":
		print("Building and training TF_NN.")

		model = tf.keras.models.Sequential(
			[tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation=tf.nn.relu),
			 tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(xTrain, yTrain, epochs=5)
		return model

	else:
		raise ValueError("Algorithm not recognized.")



def runModel(data, model):
	if ALGORITHM == "guesser":
		return guesserClassifier(data)

	elif ALGORITHM == "custom_net":
		print("Testing Custom_NN.")
		return model.predict(data)

	elif ALGORITHM == "tf_net":
		print("Testing TF_NN.")

		# get predictions
		preds = model.predict(data)

		# convert to only 0s and 1s
		ans = []
		for entry in preds:
			pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			pred[np.argmax(entry)] = 1
			ans.append(pred)
		return np.array(ans)

	else:
		raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
	xTest, yTest = data
	acc = 0

	for i in range(preds.shape[0]):
		if np.array_equal(preds[i], yTest[i]):
			acc = acc + 1

	accuracy = float(acc) / preds.shape[0]
	print("Classifier algorithm: %s" % ALGORITHM)
	print("Classifier accuracy: %f%%" % (accuracy * 100))

	# Getting predicted and actual vals for confusion matrix and f1 score

	pVals = []
	acVals = []

	for i in range(preds.shape[0]):
		pVals.append(np.argmax(preds[i]))
		acVals.append(np.argmax(yTest[i]))

	print("Confusion Matrix")
	print(sm.confusion_matrix(acVals, pVals))

	print("F1 Score")
	print(sm.f1_score(acVals, pVals, average='micro'))



#=========================<Main>================================================

def main():
	raw = getRawData()
	data = preprocessData(raw)
	model = trainModel(data[0])
	preds = runModel(data[1][0], model)
	evalResults(data[1], preds)



if __name__ == '__main__':
	main()
