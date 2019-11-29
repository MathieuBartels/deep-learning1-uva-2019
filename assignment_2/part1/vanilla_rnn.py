################################################################################
# MIT License
#
# Copyright (c) 2019
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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.whh = nn.Parameter(torch.zeros((num_hidden, num_hidden), dtype=torch.double))
        self.whx = nn.Parameter(torch.zeros((input_dim, num_hidden), dtype=torch.double))
        self.bh = nn.Parameter(torch.zeros((num_hidden), dtype=torch.double))
        

        self.wph = nn.Parameter(torch.zeros((num_hidden, num_classes), dtype=torch.double))
        self.bp = nn.Parameter(torch.zeros((num_classes), dtype=torch.double))

        self.sequence_length = seq_length

        self.h = torch.zeros((num_hidden), dtype=torch.double).to(device)

        self.h_list = []
        torch.nn.init.xavier_uniform_(self.whh)
        torch.nn.init.xavier_uniform_(self.whx)
        torch.nn.init.xavier_uniform_(self.wph)

    def forward(self, x):
        self.h_list = []
        h_prev = self.h
        for t in range(self.sequence_length):
            ht = (x[:, t] @ self.whx + h_prev @ self.whh.T + self.bh).tanh()
            ht.retain_grad()
            self.h_list.append(ht)
            h_prev = ht
        p = ht @ self.wph  + self.bp
        return p
