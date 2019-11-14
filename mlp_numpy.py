"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    layers = []
    n_inputs = n_inputs
    for layer_length in n_hidden:
      layer = LinearModule(n_inputs, layer_length)
      activation = LeakyReLUModule(neg_slope)
      layers.append([layer, activation])
      n_inputs = layer_length
    layer = LinearModule(n_inputs, n_classes)
    activation = SoftMaxModule()
    layers.append([layer, activation])
    self.network = layers
    self.loss = CrossEntropyModule()
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x
    for layer, activation in self.network:
      out = activation.forward(layer.forward(out))

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dout = dout
    for layer, activation in reversed(self.network):
      dout = layer.backward(activation.backward(dout))
    ########################
    # END OF YOUR CODE    #
    #######################

    return

  def update(self, learning_rate, batch_size):
    for layer, _ in self.network:
      layer.params['weight'] -= layer.grads['weight'] / batch_size * learning_rate
      layer.params['bias'] -= layer.grads['bias'] / batch_size * learning_rate
    return
      
