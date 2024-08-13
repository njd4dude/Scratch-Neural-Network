import numpy as np
from dense import Dense
from activations import Tanh
from losses import mse, mse_prime

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [Dense(2, 3), Tanh(), Dense(3, 1), Tanh()]


# train
def train():
    epochs = 10000
    learning_rate = 0.1

    for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            # forward
            output = x
            # propogate through the network forwards to get the output and the output serves as input for the next layer
            for layer in network:
                output = layer.forward(output)
                # print(f"->output: {output}")

            # error
            error += mse(y, output)

            # backward
            grad = mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
                # print(f"<-grad:\n {grad}")

        error /= len(X)
        print("%d/%d, error=%f" % (e + 1, epochs, error))

    # Save the model
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            layer.save(f"layer_{i}")


# train it here.. turn on/off train function
# train()


# test
# load trained weights and biases
for i, layer in enumerate(network):
    if isinstance(layer, Dense):
        layer.load(f"layer_{i}")


def predict(x):
    output = x
    for layer in network:
        output = layer.forward(output)
    return output


for x in X:
    print("\n----predicting.....\n")
    print(f"Input: {x}, Predicted: {predict(x)}")
