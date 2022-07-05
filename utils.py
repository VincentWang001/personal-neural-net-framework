import numpy as np

#vector of length 10, all 0s with i as 1, corresponding to digit i as the correct digit
def one_hot(i):
    ret = np.zeros(10)
    ret[i] = 1
    return ret

#sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#derivative of sigmoid function  
def d_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

#loss function
def loss(actual, expected):
    return sum(0.5 * (actual - expected) ** 2)