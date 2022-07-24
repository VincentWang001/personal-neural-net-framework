from NeuralNetworkFramework import NeuralNetworkFramework

import numpy as np
import utils
from keras.datasets import mnist
from matplotlib import pyplot as plt


nn = NeuralNetworkFramework()

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(len(train_X))
# input = train_X[0].flatten()
loss_arr = []
for epoch in range(10):
    batch = []
    batch_digits = []
    # batch_indices = utils.get_minibatch(nn.batch_size)
    batch_indices = [0]
    for i in batch_indices:
        batch.append(train_X[i].flatten())
        batch_digits.append(train_y[i])
    batch = np.vstack(batch) #64 x 784
    dW_arr = []
    db_arr = []
    losses = []
    for i in range(nn.batch_size):
        input = batch[i]
        correct_digit = batch_digits[i]
        dW, db, loss = nn.train(input, correct_digit)
        dW_arr.append(dW)
        db_arr.append(db)
        losses.append(loss)
            
    dW_batch = {}
    db_batch = {}

    for i in range(1, nn.num_layers + 1):
        dW_batch[i] = np.mean([dW[i] for dW in dW_arr], 0)
        db_batch[i] = np.mean([db[i] for db in db_arr], 0)
    if epoch != 0:
        loss_arr.append(np.average(losses))
        print("loss:", loss_arr[-1])
    nn.update_weights_and_biases(dW_batch, db_batch)

plt.plot(loss_arr)
plt.show()