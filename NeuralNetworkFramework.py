import numpy as np
import utils

#3 layers, 784 -> 16 -> 16 -> 10
class NeuralNetworkFramework:
    def __init__(self):
        #weights
        self.W = {
            1 : np.random.rand(16, 784) / 100,
            2: np.random.rand(16, 16) / 100,
            3: np.random.rand(10, 16) / 100
        }

        #biases
        self.b = {
            1: np.random.rand(16) / 100,
            2: np.random.rand(16) / 100,
            3: np.random.rand(10) / 100
        }
        self.learning_rate = 0.01
        self.num_layers = 3
        self.batch_size = 1
        self.activation = utils.relu
        self.d_activation = utils.d_relu
    
    #for 1 image
    def train(self, input_vec, digit):
        y = utils.one_hot(digit) # expected correct classification
        #forward pass
        z = {}
        a = {}
        z[1] = self.W[1] @ input_vec + self.b[1] # 16 x 1
        a[1] = self.activation(z[1]) # 16 x 1
        z[2] = self.W[2] @ a[1] + self.b[2] # 16 x 1
        a[2] = self.activation(z[2]) # 16 x 1
        z[3] = self.W[3] @ a[2] + self.b[3] # 10 x 1
        a[3] = z[3]#self.activation(z[3]) # 10 x 1, output
        print('a[3]: ', a[3])
        l = utils.loss(a[3], y)
        #backward pass
        dz = {}
        da = {}
        db = {}
        dW = {}

        #layer 3
        da[3] = a[3] - y #dJ/do
        dz[3] = da[3]# * self.d_activation(z[3]) #dJ/do * do/dz[3]
        db[3] = dz[3]
        dW[3] = np.outer(dz[3], a[2])

        #layer 2
        da[2] = dW[3].T @ dz[3]
        dz[2] = da[2] * self.d_activation(z[2])
        db[2] = dz[2]
        dW[2] = np.outer(dz[2], a[1])

        #layer 1
        da[1] = dW[2].T @ dz[2]
        dz[1] = da[1] * self.d_activation(z[1])
        db[1] = dz[1]
        dW[1] = np.outer(dz[1], input_vec)
        print("dW[3]: ", dW[3])
        #print("dW[2]: ", dW[2])
        #print("db[3]:", db[3])
        print("W[3]:", self.W[3])
        return (dW, db, l)
    
    def update_weights_and_biases(self, dW, db):
        for i in range(1, self.num_layers + 1):
            self.W[i] -= self.learning_rate * dW[i]
            self.b[i] -= self.learning_rate * db[i]
