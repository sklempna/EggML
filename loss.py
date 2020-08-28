from abc import ABC, abstractmethod
import numpy as np

from EggML import autodiff as ad


class LossObject(ABC):
    """
    An abstract class defining an interface for loss functions
    """
    def __init__(self, model_fcts):
        """
        Parameters: 
        model_fcts - list of autodiff Functions that define the output
        of a model
        """
        self.target_dim = len(model_fcts)
        self.target = ad.Vector(self.target_dim)
        self.loss_function = None

    def get_function(self):
        """
        Returns the composition of the loss function and model 
        functions as an autodiff Function
        """
        return self.loss_function
        

    def set_target(self, value_list):
        """
        Set the target value to allow computation of loss.
        """
        self.target.set_value(value_list)


class MSE(LossObject):
    """
    Implements the means squared error
    """
    def __init__(self, model_fcts):
        super().__init__(model_fcts)
        y_hat_minus_y = [ad.Function('-', [model_fcts[i], self.target.variables[i]]) for i in range(self.target_dim)]
        squared = [ad.Function('*', [y_hat_minus_y[i], y_hat_minus_y[i]]) for i in range(self.target_dim)]
        self.loss_function = ad.Function('+', squared)
        

class CatCrossEntropy(LossObject):
    """
    Implements the categorical crossentropy with softmax normalization
    """
    def __init__(self, model_fcts):
        super().__init__(model_fcts)
        exp_model_fcts = [ad.Function('exp', [f]) for f in model_fcts]
        sum_exp = ad.Function('+', exp_model_fcts)
        softmax_out = [ad.Function('/', [expf, sum_exp]) for expf in exp_model_fcts] 
        log_y_hat = [ad.Function('log', [f]) for f in softmax_out]
        y_log_y_hat = [ad.Function('*', [self.target[i], log_y_hat[i]]) for i in self.target_dim]
        self.loss_function = ad.Function('+', y_log_y_hat)



