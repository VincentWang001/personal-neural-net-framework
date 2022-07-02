import numpy as np

#3 layers, 784 -> 16 -> 16 -> 10
class NeuralNetworkFramework:
    def __init__(self):
        self.weights = {
            1 : np.random.rand(16, 784),
            2: np.random.rand(16, 16),
            3: np.random.rand(16, 10)
        }
        self.biases = {
            1: np.random.rand(16),
            2: np.random.rand(16),
            3: np.random.rand(10)
        }

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def loss(actual, expected):
        return 0.5 * (actual - expected) ** 2