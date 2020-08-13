import math
import numpy as np

def logistic(x):
	return 1/(1 + math.exp(-x))


def logistic_prime(x):
	math.exp(x)/(1 + math.exp(x))**2


class BasicNet:
	def __init__(self, num_of_layers, neurons_per_layer):

		self.num_of_layers = num_of_layers
		self.neurons_per_layer = neurons_per_layer

		self.weights = [[]]
		self.bias = [[]]


		for l in range(1, self.num_of_layers, 1):
			self.weights.append([np.zeros((self.neurons_per_layer[l-1]))]*self.neurons_per_layer[l])
			self.bias.append(np.zeros(self.neurons_per_layer[l]))

	def initialize_weights(self, decimal_places = 100):
		for l in range(1, self.num_of_layers, 1):
			self.weights[l] = [(np.random.rand(self.neurons_per_layer[l-1])- 0.5)*1.9 for j in range(self.neurons_per_layer[l])]

		if(decimal_places != 100):
				for l in range(1, self.num_of_layers, 1):
					for j in range(self.neurons_per_layer[l]):
						self.weights[l][j] = np.array(list(map(lambda x : round(x,decimal_places), self.weights[l][j])))

	def initialize_bias(self, decimal_places = 100):
		for l in range(1, self.num_of_layers, 1):
			self.bias[l] = (np.random.rand(self.neurons_per_layer[l]) - 0.5)*1.9
		if(decimal_places != 100):
			for l in range(1, self.num_of_layers, 1):
				self.bias[l] = np.array(list(map(lambda x: round(x,decimal_places), self.bias[l])))


	def weights_from_neuron(self,l,k):
		return np.array([self.weights[l+1][j][k] for j in range(self.neurons_per_layer[l+1])])


	def forward_pass(self, input_vec, decimal_places = 100):
		net = [[]]
		act = []

		act.append(input_vec)
		for l in range(1, self.num_of_layers, 1):
			net.append([np.dot(self.weights[l][j], act[l-1]) + self.bias[l][j] for j in range(self.neurons_per_layer[l])])
			act.append(np.array(list(map(logistic,net[l]))))
		return net, act

	def backpropagate_errors(self, net, act, target):
		errors = [[]]*self.num_of_layers
		errors[self.num_of_layers - 1] = (act[self.num_of_layers - 1] - target)*np.array(list(map(logistic_prime, net[self.num_of_layers - 1])))
		for l in range(self.num_of_layers - 2, 0, -1):
			errors[l] = np.array([np.dot(errors[l+1], self.weights_from_neuron(l,k))*logistic_prime(net[l][k]) for k in range(self.neurons_per_layer[l])])

		return errors
