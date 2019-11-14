"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    
    padding = 1
    conv1 = nn.Conv2d(n_channels, 64, (3,3), padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
    maxpool1 = nn.MaxPool2d((3,3), stride=2, padding=padding, dilation=1, return_indices=False, ceil_mode=False)

    layer1 = [conv1, maxpool1, nn.BatchNorm2d(64), nn.ReLU()]

    conv2 = nn.Conv2d(64, 128, (3,3))
    maxpool2 = nn.MaxPool2d((3,3), stride=2, padding=padding, dilation=1, return_indices=False, ceil_mode=False)

    layer2 = [conv2, maxpool2, nn.BatchNorm2d(128), nn.ReLU()]

    conv3a = nn.Conv2d(128, 256, (3,3), padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
    conv3b = nn.Conv2d(256,256, (3,3), padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
    maxpool3 = nn.MaxPool2d((3,3), stride=2, padding=padding, dilation=1, return_indices=False, ceil_mode=False)

    layer3 = [conv3a, conv3b, maxpool3, nn.BatchNorm2d(256), nn.ReLU()]

    conv4a = nn.Conv2d(256, 512, (3,3), padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
    conv4b = nn.Conv2d(512,  512, (3,3), padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
    maxpool4 = nn.MaxPool2d((3,3), stride=2, padding=padding, dilation=1, return_indices=False, ceil_mode=False)

    layer4 = [conv4a, conv4b, maxpool4, nn.BatchNorm2d(512), nn.ReLU()]

    conv5a = nn.Conv2d(512, 512, (3,3), padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
    conv5b = nn.Conv2d(512,  512, (3,3), padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros')
    maxpool5 = nn.MaxPool2d((3,3), padding=padding, dilation=1, return_indices=False, ceil_mode=False)

    layer5 = [conv5a, conv5b, maxpool5, nn.BatchNorm2d(512), nn.ReLU()]
    linear = nn.Linear(512,10)
    flat = nn.modules.Flatten()
    self.network = nn.ModuleList([*layer1, *layer2, *layer3, *layer4, *layer5,  flat, linear])
    print(self.network)

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
      # print(layer, out.shape)
      out = layer(out)
      
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
