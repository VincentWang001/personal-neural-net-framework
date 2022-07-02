import numpy as np

#3 layers, 784 -> 16 -> 16 -> 10
class NeuralNetworkFramework:
    def __init__(self):
        self.weights = {

        }
        self.biases = {

        }

    def loss(actual, expected):
        return 0.5 * (actual - expected) ** 2