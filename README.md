# From-scratch-NN-with-Minigrad
Implementation of mini autograd engine called Minigrad, inspired by Andrey Karpathy's Micrograd video. This engine is then used for the implementation of Neural Network from scratch.

Minigrad Engine provides class Value which can is used to enable automatic differentiation and backpropagation. It provides automatic differentiation for operations such as: addition, subtraction, multiplication, division, exponentiation, powering, negation and includes activation functions relu, sigmoid and tanh.
This Value class is then used to make Neuron, Layer and NeuralNetwork classes, which enable the making of artificial NN from scratch. 

Both Value class and NeuralNetwork class are tested and compared to their Pytorch counterparts. Repo contains a dummy dataset on which NN can be run as an example.
