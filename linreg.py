import math
import numpy as np

import autodiff

class LinearRegressionModel:

    def __init__(self, X, y):
        
        self.X = X
        self.y = y


# generate data

true_slope = 5
true_intercept = 2

X = (np.random.rand(100) - 0.5)*100

y = 5*X + +2 +np.random.normal(0,20,100)

#import matplotlib.pyplot as plt
#plt.plot(X,y, 'ro')
#plt.show()

# initialize values





def gradient_descent(w,b,X,y, alpha):

    w = autodiff.Variable('w', w)
    b = autodiff.Variable('b', b)
    Xs = [autodiff.Variable('', elem) for elem in X]
    ys = [autodiff.Variable('', elem) for elem in y]
    a = autodiff.Variable('', -1)

    minus_ys = [autodiff.Function('*',[a,y]) for y in ys]

    fs = []
    for i in range(len(Xs)):
        f_1 = autodiff.Function('*',[w, Xs[i]])
        f_2 = autodiff.Function('+', [f_1, b])
        f_3 = autodiff.Function('+', [f_2, minus_ys[i]])
        f_4 = autodiff.Function('*', [f_3, f_3])
        fs.append(f_4)

    f_5 = autodiff.Function('+', fs)
    MSE = autodiff.Function('*', [autodiff.Variable('', 0.01), f_5])

    new_w = w.get_value() - alpha * MSE.get_derivative('w')
    new_b = b.get_value() - alpha * MSE.get_derivative('b')

    return new_w, new_b, MSE.get_value()


w = 0
b = 0

for i in range(100):

    w, b, MSE = gradient_descent(w,b,X,y,0.001)
    print(f'iteration {i}: w is {w} and b is {b} \n MSE is {MSE}\n')
