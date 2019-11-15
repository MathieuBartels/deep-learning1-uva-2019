"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt

import torch.nn as nn
import torch

import torchvision.transforms as transforms

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  predicted = torch.max(predictions, 1)[1]
  accuracy = (predicted == targets).sum().item()/ targets.size(0)

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope
  
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  eval_freq = FLAGS.eval_freq
  batch_size = FLAGS.batch_size
  learning_rate= FLAGS.learning_rate
  feature_length = 3072
  #### FLAGS #####

    #### DATA ######
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)
  train = cifar10['train']

  train_images, train_labels = train._images, train._labels
  train_images = torch.from_numpy(train_images.reshape(train_images.shape[0], feature_length))
  train_labels = torch.max(torch.from_numpy(train_labels).long(), 1)[1]

  test_images, test_labels = cifar10['test'].images, cifar10['test'].labels 

  test_images = test_images.reshape(cifar10['test'].num_examples, feature_length)
  test_images = torch.from_numpy(test_images).cuda()
  
  test_labels = torch.max(torch.from_numpy(test_labels).long(), 1)[1].cuda()
  
  #### NEURAL NET ######
  mlp = MLP(feature_length, dnn_hidden_units, 10, neg_slope).cuda()
  # optimizer = torch.optim.Adam(mlp.parameters(),  lr=learning_rate, weight_decay=1e-6)
  optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
  loss = nn.CrossEntropyLoss()

  #### Plot data storage #####
  step_idx = []
  train_acc = []
  train_err = []
  test_acc = []
  test_err = []
  #### Step ####
  for step in range(FLAGS.max_steps):
    optimizer.zero_grad()
    mlp.train()
    ### data ####
    images, labels = train.next_batch(batch_size)
    images = torch.from_numpy(images.reshape(batch_size, feature_length)).cuda()
    labels = torch.from_numpy(labels).long().cuda()

    ### forwards and backwards ####
    prediction = mlp.forward(images)
    out = loss(prediction, torch.max(labels, 1)[1])
    out.backward()
    optimizer.step()

    with torch.no_grad():
      if not step % eval_freq:
        mlp.eval()
        step_idx.append(step)

        # prediction = mlp.forward(images)
        # train_accuracy = accuracy(prediction, torch.max(labels, 1)[1])
        # train_out = loss(prediction, torch.max(labels, 1)[1]).item()
        
        del out

        prediction = mlp.forward(train_images.cuda())
        train_labels = train_labels.cuda()
        train_accuracy = accuracy(prediction, train_labels)
        train_out = loss(prediction, train_labels).item()

        train_acc.append(train_accuracy)
        train_err.append(train_out)
        
        prediction = mlp.forward(test_images)
        test_out = loss(prediction, test_labels).item()
        test_accuracy = accuracy(prediction, test_labels)
        

        test_acc.append(test_accuracy)
        test_err.append(test_out)
        print('train error: ', train_out, ' train accuracy: ', train_accuracy,
          ' validation error: ', test_out, ' validation accuracy: ', test_accuracy)
  print(step_idx)
  print(train_acc)
  print(train_err)
  print(test_acc)
  print(test_err)
  # plt.figure(0)
  # plt.plot(step_idx, train_acc, label='train accuracy')
  # plt.plot(step_idx, test_acc, label='test accuracy')
  # plt.legend()
  # plt.title('Train and Test accuracy')
  # plt.figure(1)
  # plt.plot(step_idx, train_err, label='train error')
  # plt.plot(step_idx, test_err, label='test error')
  # plt.legend()
  # plt.title('Train and Test error')
  
  plt.show()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()