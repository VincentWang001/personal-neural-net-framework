import numpy as np
from keras.datasets import mnist

#3 layers, 784 -> 16 -> 16 -> 10
class NeuralNetworkFramework:
    def __init__(self):
        #weights
        self.W = {
            1 : np.random.rand(16, 784),
            2: np.random.rand(16, 16),
            3: np.random.rand(10, 16)
        }

        #biases
        self.b = {
            1: np.random.rand(16),
            2: np.random.rand(16),
            3: np.random.rand(10)
        }
        self.learning_rate = 100
        self.num_layers = 3
        self.batch_size = 64
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
    
    #for 1 image
    def train(self, input_vec, digit):
        y = self.one_hot(digit) # expected correct classification
        #forward pass
        z = {}
        a = {}
        z[1] = self.W[1] @ input_vec + self.b[1] # 16 x 1
        a[1] = self.sigmoid(z[1]) # 16 x 1
        z[2] = self.W[2] @ a[1] + self.b[2] # 16 x 1
        a[2] = self.sigmoid(z[2]) # 16 x 1
        z[3] = self.W[3] @ a[2] + self.b[3] # 10 x 1
        a[3] = self.sigmoid(z[3]) # 10 x 1, output
        l = self.loss(a[3], y)
        print("loss:", l)
        #backward pass
        dz = {}
        da = {}
        db = {}
        dW = {}

        #layer 3
        da[3] = a[3] - y #dJ/do
        dz[3] = da[3] * self.d_sigmoid(z[3]) #dJ/do * do/dz[3]
        db[3] = dz[3]
        dW[3] = np.outer(dz[3], a[2])

        #layer 2
        da[2] = dW[3].T @ dz[3]
        dz[2] = da[2] * self.d_sigmoid(z[2])
        db[2] = dz[2]
        dW[2] = np.outer(dz[2], a[1])

        #layer 1
        da[1] = dW[2].T @ dz[2]
        dz[1] = da[1] * self.d_sigmoid(z[1])
        db[1] = dz[1]
        dW[1] = np.outer(dz[1], input_vec)

        return (dW, db)
    
    #returns a length 10 vector of all 0s except the ith element, signifying that digit i is the correct classification
    def one_hot(self, i):
        ret = np.zeros(10)
        ret[i] = 1
        return ret
    
    def update_weights_and_biases(self, dW, db):
        for i in range(1, self.num_layers + 1):
            self.W[i] -= self.learning_rate * dW[i]
            self.b[i] -= self.learning_rate * db[i]
    
    #sigmoid function
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    #derivative of sigmoid function  
    def d_sigmoid(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    #loss function
    def loss(self, actual, expected):
        return sum(0.5 * (actual - expected) ** 2)


from matplotlib import pyplot as plt

nn = NeuralNetworkFramework()

#print(nn.train_X[0].flatten().shape)
input = nn.train_X[0].flatten()
for i in range(30):
    dW, db = nn.train(input, 5)
    #for i in range(1, 4):
        #print(db[i].shape)
        #print(nn.b[i].shape)
    nn.update_weights_and_biases(dW, db)
# for i in range(9):  
#     plt.subplot(330 + 1 + i)
#     plt.imshow(nn.train_X[i], cmap=plt.get_cmap('gray'))
#     plt.show()