import numpy as np

from EggML import autodiff as ad
from EggML import loss
from EggML import model

class SGD:
    def __init__(self, model_, loss_='mse'):
        self.model = model_
        self.model_fcts = self.model.get_model_fcts()
        if loss_=='mse':
            self.loss_obj = loss.MSE(self.model_fcts)

    def update_model(self, Xs, ys, lr = 0.1, online=True):
        summed_derivatives = np.array([0.0]*len(self.model.get_parameters_as_list()))
        for x,y in list(zip(Xs,ys)):
            self.model.set_input(x)
            self.loss_obj.set_target(y)


            params = self.model.get_parameters_as_list()
            loss_function = self.loss_obj.get_function()
            derivatives = []
            for p in params:
                derivatives.append(loss_function.get_derivative(p))
            if online:
                old_values = np.array([p.get_value() for p in params])
                new_values = old_values - lr * np.array(derivatives)
                self.model.set_parameters_from_list(new_values)
            else:
                summed_derivatives += np.array(derivatives)
        if not online:
            old_values = np.array([p.get_value() for p in params])
            new_values = old_values - lr / len(Xs) * np.array(summed_derivatives)
            self.model.set_parameters_from_list(new_values)

    def evaluate_model(self, Xs, ys, loss_='mse'):
        error = 0.0
        for x,y in list(zip(Xs,ys)):
            self.model.set_input(x)
            self.loss_obj.set_target(y)
            loss_function = self.loss_obj.get_function()
            error += loss_function.get_value()
        return error / len(Xs)



        

    

