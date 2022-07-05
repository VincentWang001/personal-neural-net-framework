from NeuralNetworkFramework import NeuralNetworkFramework

import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

nn = NeuralNetworkFramework()

(train_X, train_y), (test_X, test_y) = mnist.load_data()
        
input = train_X[0].flatten()
loss_arr = []
for i in range(600):
    dW, db, loss = nn.train(input, 5)
    print("loss:", loss)
    loss_arr.append(loss)
    nn.update_weights_and_biases(dW, db)

plt.plot(loss_arr)