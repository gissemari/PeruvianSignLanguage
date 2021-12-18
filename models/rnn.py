# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:44:04 2021

@author: Joe
"""
# Standard library imports

# Third party imports
import torch

# Local imports

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):

    def __init__(self, inputSize, hiddenSize, numLayers, outputSize, dropout=0):

        super(Net, self).__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.dropoutValue = dropout
        if(dropout and numLayers>1):
            self.rnn = torch.nn.RNN(
                inputSize, hiddenSize, numLayers, dropout=dropout, batch_first=True,nonlinearity="relu")
        elif(dropout and numLayers==1):
            self.rnn = torch.nn.RNN(
                inputSize, hiddenSize, numLayers, batch_first=True,nonlinearity="relu")
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.rnn = torch.nn.RNN(
                inputSize, hiddenSize, numLayers, batch_first=True,nonlinearity="relu")
        self.layerNorm = torch.nn.LayerNorm(hiddenSize)
        self.fc = torch.nn.Linear(hiddenSize, outputSize)

    def forward(self, x):

        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(device)

        out, hidden = self.rnn(x, h0)

        if(self.dropoutValue and self.numLayers==1):
            out = self.dropout(out)

        out = self.layerNorm(out)

        out = self.fc(out[:, -1, :])

        return out, hidden
