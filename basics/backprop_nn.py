from random import seed
from random import random
from math import exp
# network initializing

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    
    return network

# initializing testing

seed(1)

network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)
    
"""
Forward propagation
Neuron activation

First step is to calculate the activation of one neuron given an input.
The inputh could be a fow from our training dataset, as in the case of
the hidden layer. It may also be the outputs from each neuron in the
hidden layer, in the case of the output layer.

activation = sum(weight_i * input_i) + bias

Where weight is a network weight, input is an input, i is the index of
a weight or an input and bias is a special weight that has no input to
multiply with
"""

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
        
    return activation

"""
Neuron transfer

Once a neuron is activated, we need to transfer the activation to see 
what the nwuron output actually is. Different transfer functions can 
be used. It is traditional to use the sigmoid activation function, but 
you can also use the tanh function to transfer outputs. More recently,
the rectifier transfer function has been popular with large deep 
learning networks.

We can transfer an activation function using the sigmoid function
as follows:
output = 1 / (1 + e^(-activation))
"""

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

"""
Forward propagation
"""