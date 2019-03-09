import numpy as np
from matplotlib import pyplot as plt

input = np.linspace(-10,10,100)

# crunches nums between 1 and 0

def sigmoid(x):
    return 1/(1+np.exp(-x))



def sigmoid_def(x):
    return sigmoid(x)*(1-sigmoid(x))


#plt.plot(input, sigmoid(input), c='r')

