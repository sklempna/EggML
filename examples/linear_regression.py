import os
import sys

# this script should be executed from the directory containing EggML
sys.path.append(os.getcwd())

import numpy as np

from EggML import model
from EggML import optimize

# generate random data
Xs = []
for i in range(10):
    Xs.append(np.random.randint(-10,10,3))

# set true parameters of the linear map 
true_w = np.array([1,2,3])
true_b = -1

# calculate target values
ys = [[np.dot(true_w, x) + true_b] for x in Xs]

model = model.Perceptron(3, 'linear')
model.initialize()
optimizer = optimize.SGD(model)
mse = optimizer.evaluate_model(Xs, ys)
weights = [var.get_value() for var in model.get_parameters_as_list()]
print(f'\ninitial mse is equal to {mse}')
print(f'initial parameters of the linear model are {weights}\n')

for i in range(10):
    optimizer.update_model(Xs, ys, lr = 0.001)
    mse = optimizer.evaluate_model(Xs, ys)
    weights = [var.get_value() for var in model.get_parameters_as_list()]
    print(f'After generation {i+1} the mse is equal to {mse}')
    print(f'weights are {weights}\n')


