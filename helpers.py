import math

# activation functions

def logistic(x):
	return 1/(1 + math.exp(-x))

def logistic_prime(x):
	return math.exp(x)/(1 + math.exp(x))**2

def tanh(x):
	return (math.exp(2*x) - 1)/(math.exp(2*x) + 1)

def tanh_prime(x):
	return 1 - tanh(x)**2

def linear(x):
	return x

def linear_prime(x):
	return 1

act_functions = {'logistic' : (logistic, logistic_prime),
					'tanh' : (tanh, tanh_prime),
					'linear' : (linear, linear_prime)}
