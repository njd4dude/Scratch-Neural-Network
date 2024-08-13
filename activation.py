from base import Layer
import numpy as np


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        # print("\n-------------Forward step(Activiation) -------\n")
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # print("\n-------------Backward step(Activation) -------\n")
        return np.multiply(output_gradient, self.activation_prime(self.input))
