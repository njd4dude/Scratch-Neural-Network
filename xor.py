import numpy as np
from dense import Dense
from activations import Tanh
from losses import mse, mse_prime

# X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
# Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# multiply by 3 dataset
# X = np.reshape([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], (10, 1, 1))
# Y = np.reshape([[3], [6], [9], [12], [15], [18], [21], [24], [27], [30]], (10, 1, 1))
# Define X with values from 1 to 20
X = np.reshape(
    [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15],
        [16],
        [17],
    ],
    (17, 1, 1),
)

# Define Y as values from 1 to 20 multiplied by 3
Y = np.reshape(
    [
        [3],
        [6],
        [9],
        [12],
        [15],
        [18],
        [21],
        [24],
        [27],
        [30],
        [33],
        [36],
        [39],
        [42],
        [45],
        [48],
        [51],
    ],
    (17, 1, 1),
)

network = [Dense(1, 3), Dense(3, 1)]


# train
def train():
    epochs = 100
    learning_rate = 0.0001 # observation: it seems if the learning rate is too high it just caputres too much noise and causes a Nan value 

    for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            print(f"\nInput: {x}, Expected: {y}")
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
                print(f"layer: {layer}")
                grad = layer.backward(grad, learning_rate)
                # print(f"<-grad:\n {grad}")

        error /= len(X)
        print()
        print("%d/%d, error=%f" % (e + 1, epochs, error))

    # Save the model
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            layer.save(f"layer_{i}")


# train it here.. turn on/off train function
train()


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


predictX = np.reshape(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]], (12, 1, 1)
)

for x in predictX:
    print("\n----predicting.....\n")
    print(f"Input: {x}, Predicted: {predict(x)}")
