from value import Value
import random
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sys
from time import time
sys.setrecursionlimit(20000)

class Neuron(object):

    def __init__(self, number_inputs, activation_function):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(number_inputs)]
        self.bias = Value(0)
        
        if activation_function == 'linear':
            self.activation_function = lambda x: x
        elif activation_function == 'relu':
            self.activation_function = Value.relu
        elif activation_function == 'sigmoid':
            self.activation_function = Value.sigmoid
        elif activation_function == 'tanh':
            self.activation_function = Value.tanh
        

    def __call__(self, x):
        val = self.activation_function(sum(weight * data_value for weight, data_value in zip(self.weights, x)) + self.bias)
        return val


class Layer(object):

    def __init__(self, number_inputs, number_outputs, activation_function):
        self.number_inputs = number_inputs
        self.number_outputs = number_outputs
        self.neurons = [Neuron(number_inputs, activation_function) for i in range(number_outputs)]
    
    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        if len(out) == 1:
            out = out[0]
        return out

        
class NeuralNetwork(object):
    '''
    Neural Network object which consists of multiple Layers of Neurons.
    Inputs:
        -layer_sizes (list of ints): number of neurons in each layer
        -activation_functions (list of strings): activation function for each layer
    '''
    def __init__(self, input_length, output_length, layer_sizes, activation_functions, loss_function, learning_rate):
        
        layer_sizes += [output_length]
        layers_size = []
        input_num = input_length
        for i in range(len(layer_sizes)):
            output_num = layer_sizes[i]
            layers_size.append((input_num, output_num))
            input_num = output_num

        self.network = [Layer(input_num, output_num, act_func) for (input_num, output_num), act_func in zip(layers_size, activation_functions)]

        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def forward(self, xs):
        
        preds = []

        for x in xs:
            for layer in self.network:
                x = layer(x)
            preds.append(x)
        
        return preds
    
    def calculate_loss(self, preds, labels):
        #print([pred.data for pred in preds[:10]])
        #print(labels[:10])
        #input()
        if self.loss_function == 'quadratic_loss':
            loss = sum((pred-label)**2 for pred, label in zip(preds, labels))
        else:
            raise Exception("Only valid value for loss is 'quadratic_loss'")
        
        return loss
    
    def optimizer_step(self):
        for layer in self.network:
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    weight.data -= self.learning_rate * weight.grad
                neuron.bias.data -= self.learning_rate * neuron.bias.grad

    def zero_grad(self):
        for layer in self.network:
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    weight.grad = 0
                neuron.bias.grad = 0
    
    def train(self, features, labels, num_iter):
        losses = []

        for _ in range(num_iter):

            # forward pass
            preds = self.forward(features) 

            # get loss
            loss = self.calculate_loss(preds, labels)

            # backtrack
            loss.backprop()

            # update
            self.optimizer_step()

            #zero grad
            self.zero_grad()

            losses.append(loss)
        
        return losses

def main():

    

    # loading data
    data = pd.read_csv('dummy.csv')
    features = data.loc[:, data.columns != 'res']
    features = np.array(features).astype('float')
    features = (features).tolist()
    labels = data.loc[:, 'res']
    labels = np.array(labels)
    labels = labels.tolist()
    

    # constructing a nn with given architecture
    layer_sizes = [5]
    activation_functions = ['relu', 'linear']
    nn = NeuralNetwork(
                    layer_sizes=layer_sizes, 
                    input_length=len(features[0]), 
                    output_length=1, 
                    activation_functions=activation_functions, 
                    loss_function='quadratic_loss', 
                    learning_rate=0.001
                )
    
    start_time = time()

    # training
    losses = nn.train(features, labels, 1000)

    losses = [loss.data for loss in losses]
    preds = nn.forward(features)
    preds = [pred.data for pred in preds]

    # printing predictions and labels for given dataset
    for pred, lab in zip(preds, labels):
        print(f'pred = {pred}, label = {lab}')
    
    # showing time elapsed and losses during training
    print(f'time elapsed = {time() - start_time}')
    plt.plot(losses)
    plt.show()

main()

