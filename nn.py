import numpy as np
from enum import Enum

class ANN:
    """
    Artificial Neural Network class.

    ...
    Attributes
    ----------
    units_per_layer : int list
        List of the number of units for each layer
    num_layers : int
        Number of layers in the neural network
    activation : Activation_type
        Activation type of the hidden layers
    Ws : list of numpy multidimensional float arrays
        Weights of the model
    Bs : list of numpy float arrays
        Bias of the model

    Methods
    -------
    predict(X)
        Applies the input X to the model.
    set_weights(layer_index, unit, weights=[], bias=np.NaN)
        Sets the weights of a given layer.
    """

    class Activation_type(Enum):
        """
        Activation type enum.
        """
        sigmoid = 1
        tanh = 2
        relu = 3 
        relu_limited = 4
        mexican_hat = 5
        ceil = 6
        step = 7

    def __init__(self, units_per_layer, activations):
        """
        Populates model's parameters.

        Parameters
        ----------
        units_per_layer : int list
            The list of the number of units in each layer, from inputs to outputs
        activation : Activation_type list
            Activations of the hidden/output layers
        """
        self.units_per_layer = units_per_layer
        self.num_layers = len(units_per_layer) - 1 # input units are discounted
        self.activations = activations
        assert(len(self.activations) == self.num_layers)
        # pre-computation for mexican hat activation
        self.__mex_hat_sig = 1
        self.__mex_hat_A = 2/(np.sqrt(3*self.__mex_hat_sig)*np.power(np.pi,0.25))
        # max limit of relu limited activation
        self.__relu_limited_max = 2
        # creation of weights and bias lists
        self.Ws,self.Bs = [],[]
        for layer_index in range(self.num_layers):
            self.Ws.append(np.zeros((units_per_layer[layer_index+1], 
                                     units_per_layer[layer_index]), dtype=float))
            self.Bs.append(np.zeros((units_per_layer[layer_index+1],1), dtype=float))

    def __activation(self, z, type):
        """
        Calculates the activation according to the type.

        Parameters
        ----------
        z : numpy multidimensional float array
            Units outputs pre-activation values
        type : Activation_type
            Activation type
        
        Returns
        -------
        numpy multidimensional float array
            Units outputs after activation
        """
        if type is self.Activation_type.sigmoid:
            return 1 / (1 + np.exp(-z))
        elif type is self.Activation_type.tanh:
            return np.tanh(z)
        elif type is self.Activation_type.relu:
            return np.maximum(z, 0)
        elif type is self.Activation_type.relu_limited:
            return np.minimum(np.maximum(z, 0), self.__relu_limited_max)
        elif type is self.Activation_type.mexican_hat:
            z_div_a_squared = np.power(z/self.__mex_hat_sig, 2)
            return self.__mex_hat_A * (1 - z_div_a_squared) * np.exp(-0.5*z_div_a_squared)
        elif type is self.Activation_type.ceil:
            return np.ceil(z)
        elif type is self.Activation_type.step:
            return np.maximum(np.minimum(np.ceil(z), 1), 0)

    def predict(self, X):
        """
        Applies the input X to the model.

        Parameters
        ----------
        X : numpy multidimensional float array
            Inputs for the model

        Returns
        -------
        numpy multidimensional float array
            Output of the model
        dictionary
            Cache of z calculations and previous activations
        """
        # length of X must be equal to the number of inputs
        assert(len(X) == self.units_per_layer[0])
        
        current_X = X
        for layer_index in range(self.num_layers):
            W = self.Ws[layer_index]
            b = self.Bs[layer_index]
            activation = self.activations[layer_index]
            # calculate activation of a layer's units
            z = np.dot(W, current_X) + b
            current_X = self.__activation(z, activation)
        return current_X

    def set_weights(self, layer_index, unit, weights=[], bias=np.NaN):
        """
        Sets the weights of a given layer.

        Parameters
        ----------
        layer_index : int
            Index of the layer
        unit : int
            Index of the unit
        weights : float array
            The new weights
        bias : float
            The new bias
        """
        if len(weights) > 0:
            assert(len(weights) == len(self.Ws[layer_index][unit]))
            self.Ws[layer_index][unit] = np.array(weights, dtype=float)
        if not np.isnan(bias):
            self.Bs[layer_index][unit][0] = bias

