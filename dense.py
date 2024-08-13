import numpy as np
from base import Layer


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.number = 0

    def forward(self, input):
        # print("\n-------------Forward step(Dense) -------\n")
        self.input = input
        dotproduct = np.dot(self.weights, self.input) + self.bias

        # print(f"self.weights: {self.weights}")
        # print(f"self.input: {self.input}")

        return dotproduct

    def backward(self, output_gradient, learning_rate):
        # print("\n-------------Backward step*(Dense) -------\n")
        weights_gradient = np.dot(output_gradient, self.input.T)
        # print(
        #     f"adjust by weights gradient: {weights_gradient} = output_gradient: {output_gradient} * self.input.T: {self.input.T}"
        # )

        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        print(f"adjusted weights by subtracting: {learning_rate} * {weights_gradient}")
    
        self.bias -= learning_rate * output_gradient
        print(f"adjusted bias by subtracting: {learning_rate} * {output_gradient}")
        self.number += 1
        print(f"\nnumber of times weights adjusted and bias: {self.number}\n")
        return input_gradient  # to pass the gradient to the next previous layer to continue backpropagation

    def save(self, filename_prefix):
        # Save weights and biases to .npy files
        np.save(f"{filename_prefix}_weights.npy", self.weights)
        np.save(f"{filename_prefix}_bias.npy", self.bias)

    def load(self, filename_prefix):
        # Load weights and biases from .npy files
        self.weights = np.load(f"{filename_prefix}_weights.npy")
        self.bias = np.load(f"{filename_prefix}_bias.npy")
