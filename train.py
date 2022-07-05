from NeuralNetworkFramework import NeuralNetworkFramework

import numpy as np
import utils
from keras.datasets import mnist
from matplotlib import pyplot as plt

nn = NeuralNetworkFramework()

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(len(train_X))
input = train_X[0].flatten()
loss_arr = []
for epoch in range(5):
    batch = []
    batch_indices = utils.get_minibatch(nn.batch_size)
    for i in batch_indices:
        batch.append(train_X[i].flatten())
    batch = np.vstack(batch) #64 x 784
    print("batch size ", batch.shape)
    dW, db, loss = nn.train(input, 5)
    print("loss:", loss)
    loss_arr.append(loss)
    nn.update_weights_and_biases(dW, db)

plt.plot(loss_arr)