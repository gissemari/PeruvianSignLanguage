#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:36:16 2021

@author: joe
"""
import torch
import time

def loadFromCheckPoint(chkPntPath, net, optimizer, nEpoch):
    
    #delete this line when you use this procedure
    chkPntPath = ".\\Log\\2021_01_25-10_32_57-900_checkpoint.pt"
    
    chkpt = torch.load(chkPntPath)
    net.load_state_dict(chkpt['net_state'])
    optimizer.load_state_dict(chkpt['optimizer_state'])
    max_epochs = nEpoch
    # add thispart in netTrain
    epoch_saved = chkpt['epoch'] + 1
    for epoch in range(epoch_saved, max_epochs):
        torch.manual_seed(1 + epoch)
        # resume training as usual


def saveCheckPoint(chkPntPath, net, optimizer, epoch):

    chkPntPath = time.strftime("%Y_%m_%d-%H_%M_%S")

    fn = ".\\Logs\\" + str(chkPntPath) + str("-") + str(epoch) + "_checkpoint.pt"

    info_dict = {
        'epoch': epoch,
        'net_state': net.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    torch.save(info_dict, fn)

def saveModel(net):
    print("Saving trained model state dict ")
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    torch.save(net.state_dict(), path)