"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
  if type(m) == nn.Linear:
      torch.nn.init.normal_(m.weight, mean=0.0, std=0.0001)
      m.bias.data.fill_(0)
class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
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
    super(MLP, self).__init__()
    self.network = nn.ModuleList([])
    for layer_length in n_hidden:
      self.network.append(nn.Dropout(0.5))
      self.network.append(nn.BatchNorm1d(n_inputs))
      self.network.append(nn.Linear(n_inputs, layer_length))
      self.network.append(nn.LeakyReLU(negative_slope=neg_slope))
      n_inputs = layer_length
    # self.network.append(nn.Dropout(0.2))
    self.network.append(nn.Linear(n_inputs, n_classes))
    self.network.apply(init_weights)
    # self.network.append(nn.Softmax(dim=0))
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
    out=x
    for layer in self.network:
      out = layer(out)
      
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
