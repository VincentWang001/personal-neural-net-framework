import numpy as np

#vector of length 10, all 0s with i as 1, corresponding to digit i as the correct digit
def one_hot(i):
    ret = np.zeros(10)
    ret[i] = 1
    return ret

#sigmoid function
def sigmoid(x):
    # print('sigmoid: ', 1/(1 + np.exp(-x)))
    return 1/(1 + np.exp(-x))

#derivative of sigmoid function  
def d_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0) * 1
#loss function
def loss(actual, expected):
    return sum(0.5 * (actual - expected) ** 2)

def get_prediction(output_vec):
    return np.argmax(output_vec)

def get_minibatch(batch_size):
    return np.random.choice(range(60000), batch_size, replace=False)