import os
import sys

# this script should be executed from the directory containing EggML
sys.path.append(os.getcwd())

import numpy as np
from numpy.random import multivariate_normal as m_norm

from EggML import model
from EggML import optimize

# we generate random data drawn from two 2-dimensional
# normal distributions

mean1 = [0,0]
mean2 = [5,5]
cov1 = [[1,0],[0,1]]
cov2 = [[1,0],[0,1]]

size = 2

X1s = list(m_norm(mean1, cov1, size))
X2s = list(m_norm(mean2, cov2, size)) 

y1s = [[0]]*size
y2s = [[1]]*size

Xs = X1s + X2s
ys = y1s + y2s

# define model

model = model.MultiLayerPerceptron(2, 3, 0, 1, 'sigmoid')
model.initialize()
optimizer = optimize.SGD(model)
optimizer = optimize.SGD(model)
mse = optimizer.evaluate_model(Xs, ys)
weights = [var.get_value() for var in model.get_parameters_as_list()]
print(f'\ninitial mse is equal to {mse}')
#print(f'initial parameters of the linear model are {weights}\n')

for i in range(20):
    optimizer.update_model(Xs, ys, lr = 0.1)
    mse = optimizer.evaluate_model(Xs, ys)
    weights = [var.get_value() for var in model.get_parameters_as_list()]
    print(f'After generation {i+1} the mse is equal to {mse}')
#    print(f'weights are {weights}\n')