# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel



################################################################################

def get_accuracy(predictions, targets):
    accuracy = (predictions.max(axis=2)[1].cpu().numpy() == targets.cpu().numpy()).sum()/predictions.shape[0]
    return accuracy

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset( "LOTR.txt", config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, 
                                config.seq_length, 
                                dataset.vocab_size, 
                                config.lstm_num_hidden, 
                                config.lstm_num_layers, 
                                config.device)  # fixme

    model.to(config.device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(config.device)  # fixme  # fixme
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)  # fixme

    model.train()

    x_axis, losses, accuracies = [], [], []
    tot_step = 0
    for j in range(100):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            tot_step += 1

            # Only for time measurement of step through network
            t1 = time.time()


            pred = model.forward(batch_inputs.to(config.device))

            optimizer.zero_grad()

            #######################################################
            # Add more code here ...
            #######################################################

            loss = criterion(pred.permute(0, 2, 1)*config.temperature, batch_targets.to(config.device)) 
            accuracy =get_accuracy(pred, batch_targets.to(config.device))  # fixme

            losses.append(loss.data)
            accuracies.append(accuracy)
            x_axis.append(step)

            loss.backward()
            optimizer.step()
            # scheduler.step(loss.item())

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if tot_step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), tot_step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if tot_step % config.sample_every == 0:
                model.eval()
                # Generate some sentences by sampling from the model
                first_letter = [batch_inputs[0,0].item()]
                print(f"original string: \"{dataset.convert_to_string(first_letter + batch_targets[0,:].tolist())}\"".replace('\n',''))
                print(f"predicted string:\"{dataset.convert_to_string(first_letter + torch.max(pred,2)[1][0,:].tolist())}\"".replace('\n',''))
                model.train()

            if tot_step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

        torch.save(model.state_dict(), './models/Lorde_1000000_'+ str(config.lstm_num_layers) + "_" + str(j) + '.p')
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')


    # Self added argument for training efficiency
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--temperature', type=str, default=1, help="Temperature used for exercise 2.1.c in softmax")

    config = parser.parse_args()

    # Train the model
    train(config)
