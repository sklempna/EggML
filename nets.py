import numpy as np
import helpers


class BasicNet:


	def __init__(self, num_of_layers, neurons_per_layer, act_fct = 'tanh'):

		self.num_of_layers = num_of_layers
		# contains a range of layer indices from 1 to the last index
		self.layers = list(range(1, num_of_layers, 1))
		self.neurons = neurons_per_layer
		self.act_fct, self.act_fct_prime = helpers.act_functions[act_fct]
		self.weights = [[]]
		self.bias = [np.array([])]
		for l in self.layers:
			self.weights.append([np.zeros((self.neurons[l-1]))]*self.neurons[l])
			self.bias.append(np.zeros(self.neurons[l]))

	def initialize_weights(self):

		for l in self.layers:
			self.weights[l] = [(np.random.rand(self.neurons[l-1])- 0.5)
								* 1.9 for j in range(self.neurons[l])]

	def initialize_bias(self):

		for l in self.layers:
			self.bias[l] = (np.random.rand(self.neurons[l]) - 0.5)*1.9

	def weights_from_neuron(self,l,k):

		return np.array([self.weights[l+1][j][k] for j in range(self.neurons[l+1])])

	def forward_pass(self, input_vec):

		net = [np.array([])]
		act = []
		act.append(input_vec)
		for l in self.layers:
			net.append(np.array([np.dot(self.weights[l][j], act[l-1])
						+ self.bias[l][j] for j in range(self.neurons[l])]))
			act.append(np.array(list(map(self.act_fct,net[l]))))
		return net, act

	def backpropagate_errors(self, net, act, target):

		errors = [np.array([])]*self.num_of_layers
		errors[self.num_of_layers - 1] = (act[self.num_of_layers - 1] - target) \
										* np.array(list(map(self.act_fct_prime, \
										net[self.num_of_layers - 1])))
#		for l in range(self.num_of_layers - 2, 0, -1):
		for l in reversed(self.layers[:-1]):
			errors[l] = np.array([np.dot(errors[l+1], self.weights_from_neuron(l,k)) \
									*self.act_fct_prime(net[l][k]) \
									for k in range(self.neurons[l])])
		return errors
