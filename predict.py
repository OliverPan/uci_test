import numpy as np


def inference(data, weights1, biases1, weights2, biases2):

	layer1 = relu(np.matmul(data, weights1) + biases1)

	return np.matmul(layer1, weights2) + biases2


def relu(x_input):
	shape = x_input.shape
	if shape.__len__() == 1:
		for index in range(shape[0]):
			if x_input[index] > 0:
				continue
			else:
				x_input[index] = 0
	else:
		for index_i in range(shape[0]):
			for index_j in range(shape[1]):
				if x_input[index_i][index_j] > 0:
					continue
				else:
					x_input[index_i][index_j] = 0
	return x_input


def loadtxt(directory, bit):
	weights1 = np.loadtxt(directory + bit + "w1.txt")
	weights2 = np.loadtxt(directory + bit + "w2.txt")
	biases1 = np.loadtxt(directory + bit + "b1.txt")
	biases2 = np.loadtxt(directory + bit + "b2.txt")
	return weights1, biases1, weights2, biases2


def predict(data_input, bit):
	# data = np.array(list("10011110001111011111001110000110"), dtype=np.int32)
	data = np.array(data_input, dtype=np.int32)
	weights1, biases1, weights2, biases2 = loadtxt("./parameter/parity/leak0/", "data" + str(bit))
	temp = inference(data.T, weights1, biases1, weights2, biases2)
	if abs(temp-1) < temp:
		result = 1
	else:
		result = 0
	return result

