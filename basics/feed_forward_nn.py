import numpy as np

def sigmoid(activations):
    return 1.0 / (1.0 + np.power(np.e, -activations))

class Layer():
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.input_dim = 0 # because the input dimension is the output dimensions of the previous layer
        self.activations = np.array([])
        self.weights = np.random.randn( self.output_dim, self.input_dim)
        self.biases = np.zeros(self.output_dim)
        
    def update_dim(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.random.randn(self.output_dim, self.input_dim)

    def feedforward(self, input_layer):
        input_activations = input_layer.activations
        dot_product = np.dot(self.weights, input_activations)  
        activations = np.add(dot_product, self.biases)
        
        self.activations = sigmoid(activations)
        

class InputLayer(Layer):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.activations = np.array([])


class NeuralNetwork():
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        if len(self.layers) > 0:
            input_dim = self.layers[len(self.layers) - 1].output_dim
            layer.update_dim(input_dim)
            
        self.layers.append(layer)
        
    def feedforward(self, inputs):
        self.layers[0].activations = inputs
        for i in range(1, len(self.layers)):
            layer_i = self.layers[i]
            layer_i.feedforward(self.layers[i-1])
            
        output = self.layers[len(self.layers) - 1].activations
        
        return output
    
network = NeuralNetwork()
network.add_layer(InputLayer(3))
network.add_layer(Layer(2))
network.add_layer(Layer(1))
result = network.feedforward(np.array([1, 2, 3]))
print(result)
