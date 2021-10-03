# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:44:04 2021

@author: Joe
"""
# Standard library imports

# Third party imports
import torch

# Local imports


class Net(torch.nn.Module):

    def __init__(self, inputSize, hiddenSize, numLayers, outputSize, dropout=0):

        super(Net, self).__init__()

        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        if(dropout):
            self.rnn = torch.nn.RNN(
                inputSize, hiddenSize, numLayers, dropout=dropout, batch_first=True, nonlinearity="relu")
        else:
            self.rnn = torch.nn.RNN(
                inputSize, hiddenSize, numLayers, batch_first=True, nonlinearity="relu")

        self.fc = torch.nn.Linear(hiddenSize, outputSize)

    def forward(self, x):

        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize)

        out, hidden = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out, hidden
