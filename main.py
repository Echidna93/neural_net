import numpy as np
from matplotlib import pyplot as plt

input = np.linspace(-10,10,100)

# crunches nums between 1 and 0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

plt.plot(input, sigmoid(input), c='r')

'''

x1 -------(w1)-------|
.                    |
.                    |---------------------(Output [X]  = x1w1 + ... + xmwm + b)
.                    |
xm--------(wm)-------|  

'''
# feature set

feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])

# answers neural net wants to predict = labels

labels = np.array([[1,0,0,1,1]])

# will rearrange labels to be a 1 * 5 array

labels = labels.reshape(5,1)
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
# lr = learning rate
lr = 0.05

# epoch = num times we want to train the algorithm

for epoch in range(2000):
    inputs = feature_set

    
    # feedforward step1
    # X = x1*w1 + ..... + xm*wm + b
    # taking the dot product of each node times the weight and adding bias
    XW = np.dot(feature_set, weights) + bias

    # feedforward step2
    # crunch the numbers between 0 and 1
    z = sigmoid(XW)


    # backpropagation step 1
    # find the difference between our output and the solution
    error = z - labels

    #print(error.sum())

    # backpropagation step 2

    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num
    

    single_point = np.array([0,1,0])
    result = sigmoid(np.dot(single_point,weights) + bias)
    print(result)
