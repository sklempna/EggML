from abc import ABC, abstractmethod
import numpy as np

from EggML import autodiff as ad


class Model(ABC):
    """
    Abstract class defining an interface for models
    """
    def __init__(self):
        self.input_dim = None
        self.output_dim = None
        self.model_fcts = None

    def get_input_dimension(self):
        return self.input_dim

    def get_model_fcts(self):
        return self.model_fcts

    def get_output_dimension(self):
        return self.output_dim

    @abstractmethod
    def get_parameters_as_list(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def set_functions_as_input(self, function_list):
        pass

    @abstractmethod
    def set_input(self, value_list):
        pass

    @abstractmethod
    def set_parameters_from_list(self, value_list):
        pass


class Perceptron(Model):
    """
    Implementation of a single perceptron
    """
    def __init__(self, input_dim, activation_fct):
        self.input_dim = input_dim
        self.activation_fct = activation_fct
        self.output_dim = 1

        self.w = ad.Vector(input_dim)
        self.x = ad.Vector(input_dim)
        self.b = ad.ScalarVariable(0)
        
        wx = ad.get_scalar_product(self.w.variables, self.x.variables)
        wx_plus_b = ad.Function('+', [wx, self.b])
        if activation_fct == 'linear':
            self.model_fcts = [wx_plus_b]
        else:
            self.model_fcts = [ad.Function(activation_fct, [wx_plus_b])]

    def get_parameters_as_list(self):
        return self.w.variables + [self.b]

    def initialize(self):
        self.w.initialize()

    def set_functions_as_input(self, function_list):
        self.x = function_list

    def set_input(self, value_list):
        if len(value_list) != self.input_dim:
            raise ValueError('wrong input length')
        self.x.set_value(value_list)

    def set_parameters_from_list(self, value_list):
        if len(value_list) != self.input_dim + 1:
            raise ValueError('wrong number of parameters given')
        self.w.set_value(value_list[:-1])
        self.b.set_value(value_list[-1])


class DenseLayer(Model):
    """
    Implementation of a layer of densely conneted perceptrons
    """
    def __init__(self, input_dim, output_dim, activation_fct):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fct = activation_fct

        self.neurons = [Perceptron(input_dim, activation_fct) for i in range(output_dim)]
        self.model_fcts = []
        for neuron in self.neurons:
            self.model_fcts += neuron.get_model_fcts()

    def get_parameters_as_list(self):
        param_list = []
        for neuron in self.neurons:
            param_list += neuron.get_parameters_as_list()
        return param_list

    def initialize(self):
        for neuron in self.neurons:
            neuron.initialize()

    def set_functions_as_input(self, function_list):
        for neuron in self.neurons:
            neuron.set_functions_as_input(function_list)

    def set_input(self, value_list):
        for neuron in self.neurons:
            neuron.set_input(value_list)

    def set_parameters_from_list(self, value_list):
        if len(value_list) != (self.input_dim + 1) * self.output_dim:
            raise ValueError('wrong number of parameters given')
        od = self.output_dim
        for i,neuron in enumerate(self.neurons):
            neuron.set_parameters_from_list(value_list[i*od: (i+1)*od])


