# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:15:22 2021

@author: Joe
"""

# Standard library imports
import argparse
import time

# Third party imports
import torch
from sklearn.model_selection  import train_test_split
from pandas import read_pickle
import matplotlib.pyplot as plt

# Local imports
import models.resnetRnn as resnetRnn
from utils import LoadData
import utils.classificationPlotAndPrint as pp

torch.cuda.empty_cache()
device = torch.device("cpu")
print("############ ", device, " ############")

parser = argparse.ArgumentParser(description='Classification')

# 3D boolean
parser.add_argument('--wandb', action="store_true",
                    help='To activate wandb')

parser.add_argument('--keys_input_Path', type=str,
                    default="./Data/Dataset/readyToRun/",
                    help='relative path of keypoints input.'
                    ' Default: ./Data/Dataset/keypoints/')

parser.add_argument('--image_input_Path', type=str,
                    default="./Data/Dataset/img/",
                    help='relative path of image input.' +
                    ' Default: ./Data/Dataset/img/')

parser.add_argument('--keypoints_input_Path', type=str,
                    default="./Data/Dataset/keypoints/",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Dataset/keypoints/')

args = parser.parse_args()

# ----------------------------------------------------
# 1. create Dataset and DataLoader objects

class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, dataType="train", split=0.8):

        x, y, weight, y_labels, x_timeSteps = LoadData.getData(args.keys_input_Path)

        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=split , random_state=42)

        self.dataType = dataType
        
        try:
            fileData = read_pickle(args.keypoints_input_Path + str(x[0]) + '.pkl')
           
        except:
            print("There are no instances to train the model, please check the input path")

        self.rnn_input_size = len(fileData[0])
        self.trainSize = len(y_train)
        self.testSize = len(y_test)

        self.outputSize = len(y_labels)
        
        if self.dataType == "train":
            self.x_train = X_train
            self.y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        else:
            self.x_test = X_test
            self.y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    def __len__(self):

        if self.dataType == "train":
            return self.trainSize
        else:
            return self.testSize

    def __getitem__(self, index):

        if self.dataType == "train":
            img = LoadData.getXInfo(args.image_input_Path, self.x_train[index])
            kps = LoadData.getXInfo(args.keypoints_input_Path, self.x_train[index])
            trgts = self.y_train[index]
        else:
            img = LoadData.getXInfo(args.image_input_Path, self.x_test[index])
            kps = LoadData.getXInfo(args.keypoints_input_Path, self.x_test[index])
            trgts = self.y_test[index]
        
        image = torch.tensor(img, dtype=torch.float32).to(device)
        keypoints = torch.tensor(kps, dtype=torch.float32).to(device)

        sample = {
            'image': image,
            'keypoints': keypoints,
            'targets': trgts
            }

        return sample


def accuracy_quick(yPred, yTarget):
    # assumes model.eval()
    # en masse but quick
    n = len(yTarget)

    arg_maxs = torch.argmax(yPred, dim=1)  # collapse cols
    num_correct = torch.sum(yTarget == arg_maxs)
    acc = (num_correct * 1.0 / n)

    return acc.item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    
    split = 0.8

    data_train = SignLanguageDataset(split=split)
    data_test = SignLanguageDataset(dataType="test")

    print("Begin predict sign language")
    torch.manual_seed(1)

    # variables
    dropout = 0.0
    rnnLayers = 1
    num_classes = data_train.outputSize
    batch_size = 5
    nEpoch = 2000
    lrn_rate = 0.001
    weight_decay = 0
    epsilon = 1e-8
    hidden_size = 80

    print("data train split at: %2.2f" % split)
    print("hidden size: %d" % hidden_size)
    print("dropout: %3.2f" % dropout)
    print("batch_size: %d" % batch_size)
    print("number of epoch: %d" % nEpoch)
    print("learning rate: %f" % lrn_rate)
    print("Number of rnn layers: %d" % rnnLayers)
    print("Epsilon: %3.2f" % epsilon)
    print("weight decay: %3.2f" % weight_decay)
    print("# classes: %d" % num_classes)
    
    dataTrain = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
    dataTest = torch.utils.data.DataLoader(data_test, batch_size=data_test.testSize)

    net = resnetRnn.resnet_rnn(num_classes, data_train.rnn_input_size, hidden_size, rnnLayers, dropout).to(device)
    
    print('The number of parameter is: %d' % count_parameters(net))
    
    net.train()  # set mode
    
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lrn_rate,
                                 weight_decay=weight_decay, eps=epsilon)
    
    # In case it is necesary to recover part of the trained model from checkpoint
    # chkPntPath = ""
    # bckmod.loadFromCheckPoint(chkPntPath, net, optimizer, nEpoch)
    
    net.zero_grad()

    accEpochAcum = []
    lossEpochAcum = []
    accTestEpochAcum = []
    lossTestEpochAcum = []

    fig, axs = pp.interactivePlotConf()

    start_time = time.time()
    
    for epoch in range(0, nEpoch):
        # T.manual_seed(1 + epoch)  # recovery reproducibility

        epoch_loss = 0.0  # sum avg loss per item
        epoch_acc = 0.0

        epoch_loss_test = 0.0
        epoch_acc_test = 0.0

        start_bach_time = time.time()
        
        # TRAIN
        net.train()

        for (batch_idx, batch) in enumerate(dataTrain):

            # Get data train batch
            X_img = batch['image']  # inputs
            X_kp = batch['keypoints']  # inputs
            Y = batch['targets']

            X_img = X_img.to(device)
            X_kp = X_kp.to(device)
            YTrain = Y.to(device)

            optimizer.zero_grad()

            output = net(x_img=X_img, x_kp=X_kp)

            loss_val = loss_func(output, YTrain)
            epoch_loss += loss_val.item()  # a sum of averages
            train_acc = accuracy_quick(output, YTrain)
            epoch_acc += train_acc

            # Backward
            loss_val.backward()

            # Step
            optimizer.step()

        # TEST
        net.eval()
        for (batch_idx, batch) in enumerate(dataTest):
            
            # Get data train batch
            X_img = batch['image']  # inputs
            X_kp = batch['keypoints']  # inputs
            Y = batch['targets']

            X_img = X_img.to(device)
            X_kp = X_kp.to(device)
            yTest = Y.to(device)

            with torch.no_grad():
                ouptTest = net(x_img=X_img, x_kp=X_kp)

            loss_val_test = loss_func(ouptTest, yTest)
            epoch_loss_test += loss_val_test.item()

            test_acc = accuracy_quick(ouptTest, yTest)
            epoch_acc_test += test_acc

        # if you need to save checkpoint of the model
        # chkPntPath=""
        # bckmod.saveCheckPoint(chkPntPath, net, optimizer, nEpoch)

        lossEpoch = epoch_loss/len(dataTrain)
        accEpoch = epoch_acc / len(dataTrain)
        lossEpochAcum.append(lossEpoch)
        accEpochAcum.append(accEpoch)

        lossTestEpoch = epoch_loss_test/len(dataTrain)
        accTestEpoch = epoch_acc_test / len(dataTrain)
        lossTestEpochAcum.append(lossTestEpoch)
        accTestEpochAcum.append(accTestEpoch)

        if(epoch % 1 == 0):

            # print epoch evaluation
            pp.printEpochEval(epoch, lossEpoch, accEpoch, lossTestEpoch,
                              accTestEpoch, start_bach_time)

            pp.plotEpochEval(fig, plt, axs, epoch, lossEpochAcum, lossTestEpochAcum,
                             accEpochAcum, accTestEpochAcum, rnnLayers, num_classes,
                             batch_size, nEpoch, lrn_rate, hidden_size)

    print("Done ")
    print("Total time: %0.4f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()