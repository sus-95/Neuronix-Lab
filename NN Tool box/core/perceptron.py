import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        losses = []

        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = (linear > 0).astype(int)

            error = y - y_pred

            self.weights += self.lr * np.dot(X.T, error)
            self.bias += self.lr * np.sum(error)

            losses.append(np.mean(error**2))

        return losses

    def predict(self, X):
        return (np.dot(X, self.weights) + self.bias > 0).astype(int)