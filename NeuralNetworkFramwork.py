import numpy as np

#3 layers, 784 -> 16 -> 16 -> 10
class NeuralNetworkFramework:
    def __init__(self):
        #weights
        self.W = {
            1 : np.random.rand(16, 784),
            2: np.random.rand(16, 16),
            3: np.random.rand(16, 10)
        }

        #biases
        self.b = {
            1: np.random.rand(16),
            2: np.random.rand(16),
            3: np.random.rand(10)
        }

    
    #for 1 image
    def train(self, input):
        z = {}
        a = {}
        z[1] = self.W[1] @ input + self.b[1]
        a[1] = self.sigmoid(z[1])
        z[2] = self.W[2] @ a[1] + self.b[2]
        a[2] = self.sigmoid(z[2])
        z[3] = self.W[3] @ a[2] + self.b[3]
        a[3] = self.sigmoid(z[3])
        output = a[3]
        l = self.loss(output, self.onehot(2))

    def onehot(i):
        ret = np.zeros(10)
        ret[i] = 1
        return ret

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def loss(self, actual, expected):
        return 0.5 * (actual - expected) ** 2

nn = NeuralNetworkFramework()