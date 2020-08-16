import math
import numpy as np
import helpers


class BasicNet:


    def __init__(self, num_of_layers, neurons_per_layer, act_fct = 'tanh'):

        self.num_of_layers = num_of_layers
        # contains a range of layer indices from 1 to the last index
        self.layers = list(range(1, num_of_layers, 1))
        self.neurons = neurons_per_layer

        self.act_fcts = [helpers.act_functions['linear'][0]] + [helpers.act_functions[act_fct][0]]*(self.num_of_layers - 1)
        self.act_fcts_prime = [helpers.act_functions['linear'][1]] + [helpers.act_functions[act_fct][1]]*(self.num_of_layers - 1)

        self.weights = [[]]
        self.bias = [np.array([])]
        for l in self.layers:
            self.weights.append([np.zeros((self.neurons[l-1]))]*self.neurons[l])
            self.bias.append(np.zeros(self.neurons[l]))

    def initialize_weights(self):
        """
        Initialize weights with random numbers in (-1,1)
        """
        for l in self.layers:
            self.weights[l] = [(np.random.rand(self.neurons[l-1])- 0.5)
                                * 1.9 for j in range(self.neurons[l])]

    def initialize_bias(self):
        """
        Initialize bias with random numbers in (-1,1)
        """
        for l in self.layers:
            self.bias[l] = (np.random.rand(self.neurons[l]) - 0.5)*1.9

    def weights_from_neuron(self,l,k):
        """
        Returns an array of weights emanating from a source neuron

        Parameters:
        l (int): layer of the neuron
        k (int): position inside the layer

        Returns:
        numpy array of weights
        """
        return np.array([self.weights[l+1][j][k] for j in range(self.neurons[l+1])])

    def set_activation_function(self, l, act_fct):
        """
        Set the activation functions of a certain layer

        Parameters:
        l (int): layer
        act_fct (str): name of activation function
        """
        self.act_fcts[l] = helpers.act_functions[act_fct][0]
        self.act_fcts_prime[l] = helpers.act_functions[act_fct][1]

    def forward_pass(self, input_vec):
        """

        """
        net = [np.array([])]
        act = []
        act.append(input_vec)
        for l in self.layers:
            net.append(np.array([np.dot(self.weights[l][j], act[l-1])
                        + self.bias[l][j] for j in range(self.neurons[l])]))
            act.append(np.array(list(map(self.act_fcts[l],net[l]))))
        return net, act

    def backpropagate_errors(self, net, act, target, loss):

        # errors[l][j] contains derivative of loss-fct w.r.t the net
        # input of neuron j in layer l
        errors = [np.array([])]*self.num_of_layers

        if loss == 'crossentropy':
            # automatically use softmax in last layer!
            exp_net = np.array(list(map(math.exp, net[-1])))
            errors[-1] = exp_net / sum(exp_net) * sum(target) - target 
        elif loss == 'mse':
            p = (act[-1] - target)/self.neurons[-1]
            errors[-1] = p * np.array(list(map(self.act_fcts_prime[-1], net[-1])))
        else: 
            raise ValueError(f"unknown loss function: {loss}")

        for l in reversed(self.layers[:-1]):
            errors[l] = np.array([np.dot(errors[l+1], self.weights_from_neuron(l,k)) \
                                    *self.act_fcts_prime[l](net[l][k]) \
                                    for k in range(self.neurons[l])])
        return errors
