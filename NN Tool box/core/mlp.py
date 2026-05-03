import numpy as np

class MLP:
    def __init__(self, layers, lr=0.01, epochs=100):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.biases = []

        # Initialize weights
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i+1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        activations = [X]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            a = self.sigmoid(z)
            activations.append(a)

        return activations

    def backward(self, X, y, activations):
        deltas = [activations[-1] - y]

        # Backprop
        for i in reversed(range(len(self.weights)-1)):
            delta = deltas[-1].dot(self.weights[i+1].T) * self.sigmoid_derivative(activations[i+1])
            deltas.append(delta)

        deltas.reverse()

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * activations[i].T.dot(deltas[i])
            self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y):
        losses = []

        for _ in range(self.epochs):
            activations = self.forward(X)
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)

            self.backward(X, y, activations)

        return losses

    def predict(self, X):
        return self.forward(X)[-1]