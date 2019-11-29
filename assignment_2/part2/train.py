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
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import TextDataset
from model import TextGenerationModel

from random import randint
################################################################################
def calc_accuracy(predictions, targets):
    predicted = torch.max(predictions, 1)[1]
    targets = targets
    accuracy = (predicted == targets).sum().item()/ targets.nelement()
    return accuracy

def generate(model, n, vocab_size, device, temperature, start, multi=False):
    model.eval()
    sentence = start
    for _ in range(n):
        torch_sentence = torch.nn.functional.one_hot(torch.tensor(sentence).to(torch.int64), vocab_size).to(torch.float).to(device=device)
        out = model(torch.unsqueeze(torch_sentence, 0))
        out = (out * temperature).softmax(dim=2)
        if multi:
            sentence.append(int(torch.multinomial(out[0,-1,:],1)))
        else:
            sentence.append(int(torch.argmax(out[0,-1,:])))
    model.train()
    return sentence

def finish_sentence(model, n, seq_length, vocab_size, device, start):
    
    model.eval()
    sent = start
    for _ in range(n):
        torch_sent = torch.nn.functional.one_hot(torch.from_numpy(np.array(sent[-seq_length:])).to(torch.int64), vocab_size).to(torch.float).to(device=device)
        out = model(torch.unsqueeze(torch_sent,0))
        sent.append(int(torch.argmax(out[0,-1,:])))
    model.train()
    return sent

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length + 1)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0').to(device)  # fixme

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss() # fixme
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)  # fixme
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # fixme

    acc = []
    error = []
    steps = []
    epochs = 1
    sents = []
    taus05 = []
    taus1 = []
    taus2 = []
    sentLongs = []
    sentShorts = []
    finish_sents = []
    for _ in range(epochs):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            optimizer.zero_grad()
            batch_inputs = torch.nn.functional.one_hot(batch_inputs.to(torch.int64), dataset.vocab_size).to(torch.float).to(device=device)
            batch_targets = batch_targets.to(device)
            out = model.forward(batch_inputs).permute(0, 2, 1)



            loss = criterion(out, batch_targets)   # fixme
            accuracy = calc_accuracy(out, batch_targets)  # fixme

            loss.backward()
            optimizer.step()
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
            if step % config.print_every == 0:
                # print("pred", dataset.convert_to_string(torch.max(out, 1)[1][0].cpu().numpy()))
                # print("real", dataset.convert_to_string(batch_targets[0].cpu().numpy()))
                acc.append(accuracy)
                error.append(loss.item())
                steps.append(step)

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        int(config.train_steps), config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                sent = generate(model, 30, dataset.vocab_size, config.device, config.temperature, [randint(0, dataset.vocab_size-1)])
                tau05 = generate(model, 30, dataset.vocab_size, config.device, 0.5, [randint(0, dataset.vocab_size-1)], True)
                tau1 = generate(model, 30, dataset.vocab_size, config.device, 1, [randint(0, dataset.vocab_size-1)], True)
                tau2 = generate(model, 30, dataset.vocab_size, config.device, 2, [randint(0, dataset.vocab_size-1)], True)
                sentLong = generate(model, 45, dataset.vocab_size, config.device, config.temperature, [randint(0, dataset.vocab_size-1)], False)
                sentShort = generate(model, 15, dataset.vocab_size, config.device, config.temperature, [randint(0, dataset.vocab_size-1)], False)
                print("generated sentence", dataset.convert_to_string(sent))
                sents.append(dataset.convert_to_string(sent))
                taus05.append(dataset.convert_to_string(tau05))
                taus1.append(dataset.convert_to_string(tau1))
                taus2.append(dataset.convert_to_string(tau2))
                sentLongs.append(dataset.convert_to_string(sentLong))
                sentShorts.append(dataset.convert_to_string(sentShort))
               
                finish_sent = generate(model, 50, dataset.vocab_size, device, config.temperature, dataset.convert_to_int_array("Mathieu "), True)
                print("finished sentence", dataset.convert_to_string(finish_sent))
                finish_sents.append(dataset.convert_to_string(finish_sent))
            if step == config.train_steps:
                # If you receive a PyTorch dataplt-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                print("step is trainsteps")
                break

    print('Done training.')
    # np.savetxt('acc.npy', np.array(acc))
    # np.savetxt('error.npy', np.array(error))
    # np.savetxt('steps.npy', np.array(steps))
    with open('sentences.txt', 'w') as f:
        f.writelines(["%s\n" % item  for item in sents])

    with open('tau05.txt', 'w') as f:
        f.writelines(["%s\n" % item  for item in taus05])

    with open('tau1.txt', 'w') as f:
        f.writelines(["%s\n" % item  for item in taus1])

    with open('tau2.txt', 'w') as f:
        f.writelines(["%s\n" % item  for item in taus2])

    with open('sentLong.txt', 'w') as f:
        f.writelines(["%s\n" % item  for item in sentLongs])

    with open('sentshort.txt', 'w') as f:
        f.writelines(["%s\n" % item  for item in sentShorts])

    with open('finished_sent.txt', 'w') as f:
        f.writelines(["%s\n" % item  for item in finish_sents])

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
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-2, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    # My params
    parser.add_argument('--temperature', type=int, default=1, help='temperature of the softmax')

    config = parser.parse_args()

    # Train the model
    train(config)
