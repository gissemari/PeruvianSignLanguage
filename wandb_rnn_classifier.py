#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 22:39:51 2021

@author: joe
"""

# -*- coding: utf-8 -*-

# Standard library imports
import time
import sys
import os

# Third party imports
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import wandb
from sklearn.model_selection  import train_test_split

# Local imports
sys.path.append(os.getcwd())
import utils.video as uv
from utils import LoadData
import utils.wandbFunctions as wandbF
import utils.backupModel as bckmod
import utils.classificationPlotAndPrint as pp
import models.rnn as rnn

torch.cuda.empty_cache()
device = torch.device("cuda")
print("############ ", device, " ############")
# ----------------------------------------------------
# 1. create Dataset and DataLoader objects


class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, x, y, y_labels, bodyParts, timestepSize=17, datasetType="train", split=0.8):

        self.datasetType = datasetType

        if self.datasetType == "train":
            X_train = LoadData.getKeypointsfromIdList("./Data/Dataset/keypoints/", x, bodyParts, timestepSize)
            self.X_train = torch.tensor(X_train,dtype=torch.float32).to(device)
            self.y_train = torch.tensor(y,dtype=torch.int64).to(device)
            self.inputSize = X_train.shape[2]
            print("Train: ",self.X_train.shape, self.y_train.shape)
        else:
            X_test = LoadData.getKeypointsfromIdList("./Data/Dataset/keypoints/", x, bodyParts, timestepSize)
            self.X_test = torch.tensor(X_test,dtype=torch.float32).to(device)
            self.y_test = torch.tensor(y,dtype=torch.int64).to(device)
            self.inputSize = X_test.shape[2]
            print("Test: ",self.X_test.shape, self.y_test.shape)

        self.outputSize = len(y_labels)

        self.y_labels = y_labels

    def __len__(self):
        if self.datasetType == "train":
            return len(self.y_train)
        else:
            return len(self.y_test)

    def __getitem__(self, index):
        
        if self.datasetType == "train":
            preds = self.X_train[index]
            trgts = self.y_train[index]
        else:
            preds = self.X_test[index]
            trgts = self.y_test[index]

        sample = {
            'predictors': preds,
            'targets': trgts}

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


def compute_loss_and_acc(loss_func, net, dataloader):
    "compute average loss over the a dataset"
    net.eval()

    y_acum = torch.tensor([], dtype=torch.int64).to(device)
    output_acum = torch.tensor([], dtype=torch.float32).to(device)

    for (batch_idx, batch) in enumerate(dataloader):

        # Get data train batch
        X_kp = batch['predictors']  # inputs
        Y = batch['targets']
        Y = Y.to(device, dtype=torch.int64)

        with torch.no_grad():
            output, _ = net(X_kp)

        y_acum = torch.cat((y_acum, Y))
        output_acum = torch.cat((output_acum, output))

    lossMean = loss_func(output_acum, y_acum)
    accMean = accuracy_quick(output_acum, y_acum)

    return lossMean.to("cpu").numpy(), accMean

def main():

    ##################################################
    # 0. get started

    ##################################################
    # 1. create Dataset and DataLoader objects

    # with open(args.output_Path+'3D/X.data','rb') as f: new_data = pkl.load(f)

    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    print("Begin predict sign language")
    
    np.random.seed(1)
    torch.manual_seed(1)

    hyperparameter_defaults = dict(
        dropout = 0.25,
        num_layers = 1,
        lrn_rate = 0.0005,
        weight_decay = 0,
        epsilon = 1e-8,
        bodyParts = ["pose", "hands", "face"],
        num_classes = 10,
        hidden_size = 180,
        epoch = 1000,
        batch_size = 64
    )
    # Dataset variables
    timestepSize = 17
    split = 0.8
    
    # Access all hyperparameter values through wandb.config
    wandb.init(project='PSL', entity='joenatan30',
               config=hyperparameter_defaults)

    config = wandb.config
    
    x, y, weight, y_labels, x_timeSteps = LoadData.getData("./Data/Dataset/readyToRun/")

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=split , random_state=42, stratify=y)

    dataTrainXY = SignLanguageDataset(src, X_train, y_train, y_labels, config["bodyParts"], timestepSize=timestepSize, datasetType="train",split=split)
    dataTestXY = SignLanguageDataset(src, X_test, y_test, y_labels, config["bodyParts"], timestepSize=timestepSize,datasetType="test",split=split)

    # variables
    split = 0.8
    num_classes = dataTrainXY.outputSize

    print("data train split at: %2.2f" % split)
    print("hidden size: %d" % config["hidden_size"])
    print("batch_size: %d" % config["batch_size"])
    print("number of epoch: %d" % config["epoch"])
    print("learning rate: %f" % config["lrn_rate"])
    print("Dropout: %f" % config["dropout"])
    print("Weight decay: %f" % config["weight_decay"])
    print("epsilon: %f" % config["epsilon"])
    print("Number of layers: %d" % config["num_layers"])
    print("Body parts:", config["bodyParts"] )

    dataTrain = torch.utils.data.DataLoader(dataTrainXY, batch_size=config["batch_size"])
    dataTest = torch.utils.data.DataLoader(dataTestXY, batch_size=config["batch_size"])

    ##################################################
    # 2. create neural network
    net = rnn.Net(dataTrainXY.inputSize, config["hidden_size"],
              config["num_layers"], dataTrainXY.outputSize, dropout=config["dropout"]).to(device)

    print('The number of parameter is: %d' % count_parameters(net))

    # Wandb the network weight
    wandbF.watch(net)

    ##################################################
    # 3. train network
    net.train()  # set mode

    # loss_func = torch.nn.CrossEntropyLoss(weight=dataXY.weight)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lrn_rate"],
                                 weight_decay=config["weight_decay"],
                                 eps=config["epsilon"])

    net.zero_grad()

    accEpochAcum = []
    lossEpochAcum = []
    accTestEpochAcum = []
    lossTestEpochAcum = []

    start_time = time.time()
    start_bach_time = time.time()
    
    for epoch in range(0, config["epoch"]):
        # T.manual_seed(1 + epoch)  # recovery reproducibility  

        net.train()

        for (batch_idx, batch) in enumerate(dataTrain):

            # Get data train batch
            X = batch['predictors']  # inputs
            Y = batch['targets']
            XTrain = X.to(device)
            YTrain = Y.to(device)

            optimizer.zero_grad()

            output, hidden = net(XTrain)

            loss_val = loss_func(output, YTrain)

            # Backward
            loss_val.backward()

            # Step
            optimizer.step()

        train_loss, train_acc = compute_loss_and_acc(loss_func, net, dataTrain)
        test_loss, test_acc = compute_loss_and_acc(loss_func, net, dataTest)

        lossEpochAcum.append(train_loss)
        accEpochAcum.append(train_acc)

        lossTestEpochAcum.append(test_loss)
        accTestEpochAcum.append(test_acc)

        if(epoch % 1 == 0):

            #Log in wandb
            wandb.log({"Train_loss": train_loss,
               "Train_accuracy": train_acc,
               "Test_Loss": test_loss,
               "Test_accuracy": test_acc
               })
            
            # print epoch evaluation
            pp.printEpochEval(epoch, train_loss, train_acc, test_loss,
                              test_acc, start_bach_time)

            start_bach_time = time.time()

    print("Done ")
    print("Total time: %0.4f seconds" % (time.time() - start_time))
    ########################
    # END of the training section
    ##################################################

    # Prepare folders
    uv.createFolder("./evaluation")
    uv.createFolder("./evaluation/rnn/")
    uv.createFolder("./evaluation/rnn/classes_%d" % num_classes)
    uv.createFolder("./evaluation/rnn/classes_%d/layers_%d" % (num_classes, config["num_layers"]))
    uv.createFolder("./evaluation/rnn/classes_%d/layers_%d/lrnRt_%f" % (num_classes, config["num_layers"], config["lrn_rate"]))
    uv.createFolder("./evaluation/rnn/classes_%d/layers_%d/lrnRt_%f/batch-%d" % (num_classes, config["num_layers"], config["lrn_rate"], config["batch_size"]))
    pltSavePath = "./evaluation/rnn/classes_%d/layers_%d/lrnRt_%f/batch-%d" % (num_classes, config["num_layers"], config["lrn_rate"], config["batch_size"])
    plt.savefig(pltSavePath + '/rnn-LOSS_lrnRt-%f_batch-%d_nEpoch-%d_hidden-%d.png' % (config["lrn_rate"], config["batch_size"], config["epoch"], config["hidden_size"]))

    ##################################################
    # 4. evaluate model

    net.eval()

    src = "./Data/Keypoints/pkl/Segmented_gestures/"

    ###
    # Test Accuracy ###

    X_test = dataTestXY.X_test
    Y_test = dataTestXY.y_test

    with torch.no_grad():
        ouptTest, _ = net(X_test)

    acc = accuracy_quick(ouptTest, Y_test)
    print("=======================================")
    print("\nTest Accuracy = %0.4f" % acc)

    ###
    # Confusion matrix (CM) ###

    confusion_matrix_test = torch.zeros(num_classes, num_classes)
    confusion_matrix_train = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        # CM Test
        inputsTest = dataTestXY.X_test.to(device)
        targetTest = dataTestXY.y_test.to(device)

        outputsTest, _ = net(inputsTest)

        _, predsTest = torch.max(outputsTest, 1)

        for t, p in zip(targetTest.view(-1), predsTest.view(-1)):
            confusion_matrix_test[t.long(), p.long()] += 1

        # CM Train
        inputsTrain = dataTrainXY.X_train.to(device)
        targetTrain = dataTrainXY.y_train.to(device)

        outputsTrain, _ = net(inputsTrain)

        _, predsTrain = torch.max(outputsTrain, 1)

        for t, p in zip(targetTrain.view(-1), predsTrain.view(-1)):
            confusion_matrix_train[t.long(), p.long()] += 1

    ###
    # Plot CM Test ###

    confusion_matrix_test = confusion_matrix_test.to("cpu").numpy()

    ###
    # Plot CM Train ###

    confusion_matrix_train = confusion_matrix_train.to("cpu").numpy()

    # Send confusion matrix Test to Wandb
    wandbF.sendConfusionMatrix(targetTest.to("cpu").numpy(),
                                   predsTest.to("cpu").numpy(),
                                   list(dataTrainXY.y_labels.values()),
                                   cmTrain=False)

    # Send confusion matrix Train to Wandb
    wandbF.sendConfusionMatrix(targetTrain.to("cpu").numpy(),
                                   predsTrain.to("cpu").numpy(),
                                   list(dataTrainXY.y_labels.values()),
                                   cmTrain=True)

    ##################################################
    # 5. save model

    bckmod.saveModel(net)

    ##################################################
    # 6. make a prediction

    wandbF.finishWandb()


if __name__ == "__main__":
    main()
