import math
import numpy as np

class ScalarVariable:
    def __init__(self, value):
        self.value = value

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_derivative(self, variable):
        return int(self == variable)


class Vector:
    def __init__(self, dimension, function_vector = False):
        self.dimension = dimension
        self.variables = [ScalarVariable(0) for i in range(dimension)]
        self.function_vector = False

    def set_value(self, value_list):
        if self.function_vector:
            print("can't set values of function vector")
        else:
            for i, value in enumerate(value_list):
                self.variables[i].set_value(value)

    def get_value(self):
        return [var.get_value() for var in self.variables]

    def initialize(self):
        if self.function_vector:
            print("can't initialize function vector")
        else:
            value_list = np.random.rand(self.dimension)
            self.set_value(value_list)

    def set_functions(self, function_list):
        self.variables = function_list
        self.function_vector = True


class Function:
    def __init__(self, operator, children):
        self.operator = operator
        self.children = np.array(children)
        if operator == '-' and len(children) != 2:
            raise ValueError("'-' operator must have 2 children")
        if operator == '/' and len(children) != 2:
            raise ValueError("'/' operator must have 2 children")
        if operator == 'tanh' and len(children) != 1:
            raise ValueError("'tanh' operator must have 1 child")
        if operator == 'exp' and len(children) != 1:
            raise ValueError("'exp' operator must have 1 child")
        if operator == 'sigmoid' and len(children) != 1:
            raise ValueError("'sigmoid' operator must have 1 child")
        if operator == 'log' and len(children) != 1:
            raise ValueError("'log' operator must have 1 child")

    def children_wo_index(self, index):
        return self.children[np.arange(len(self.children)) != index]

    def prod_wo_index(self, index):
        return np.prod([child.get_value() 
                        for child in self.children_wo_index(index)])

    def get_value(self):
        if self.operator == '+':
            return sum([child.get_value() for child in self.children])
        elif self.operator == '*':
            return np.prod([child.get_value() 
                            for child in self.children])
        elif self.operator == '-':
            return self.children[0].get_value() \
                    - self.children[1].get_value()
        elif self.operator == '/':
            return self.children[0].get_value() \
                    / self.children[1].get_value()
        elif self.operator == 'tanh':
            return np.tanh(self.children[0].get_value())
        elif self.operator == 'exp':
            return math.exp(self.children[0].get_value())
        elif self.operator == 'sigmoid':
            val = self.children[0].get_value()
            return 1/(1 + math.exp(-val))
        elif self.operator == 'log':
            return math.log(self.children[0].get_value())
        
            
    def get_derivative(self, variable):
        if self.operator == '+':
            return sum([child.get_derivative(variable) 
                        for child in self.children])
        elif self.operator == '*':
            return sum([self.children[i].get_derivative(variable) 
                        * self.prod_wo_index(i) 
                            for i in range(len(self.children))])
        elif self.operator == '-':
            return self.children[0].get_derivative(variable) \
                    - self.children[1].get_derivative(variable)
        elif self.operator == '/':
            return (self.children[0].get_derivative(variable) \
                    * self.children[1].get_value() \
                    - self.children[0].get_value() \
                    * self.children[1].get_derivative(variable)) \
                    / self.children[1].get_value() ** 2
        elif self.operator == 'tanh':
            inner = self.children[0]
            d = inner.get_derivative(variable)
            return (1 - np.tanh(inner.get_value())**2) * d
        elif self.operator == 'exp':
            inner = self.children[0]
            d = inner.get_derivative(variable)
            return math.exp(inner.get_value()) * d
        elif self.operator == 'sigmoid':
            inner = self.children[0]
            d = inner.get_derivative(variable)
            val = math.exp(inner.get_value())
            return d * val/(1 + val)**2
        elif self.operator == 'log':
            inner = self.children[0].get_derivative(variable)
            outer = 1 / self.children[0].get_value()
            return inner * outer

def get_scalar_product(list1, list2):
    """
    Given two lists of ScalarVariables or Functions of the same length
    returns an autodiff Function computing their scalar product
    """
    if len(list1) != len(list2):
        raise ValueError("""can't construct scalar product from \
                            lists of different lengths""")
    return Function('+', [Function('*', 
                    [list1[i], list2[i]]) 
                    for i in range(len(list1))])        
