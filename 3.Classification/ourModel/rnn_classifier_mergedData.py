# -*- coding: utf-8 -*-
"""
@author: Joe

"""
# -*- coding: utf-8 -*-

# Standard library imports
import argparse
import time
import os
import sys
from pathlib import Path

# Third party imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
import pandas as pd

# Local imports
sys.path.append(os.getcwd())
import utils.video as uv
from utils import LoadData
import utils.wandbFunctions as wandbF
import utils.backupModel as bckmod
import utils.classificationPlotAndPrint as pp
import models.rnn as rnn

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("############ ", device, " ############")

# ----------------------------------------------------
# 1. create Dataset and DataLoader objects


class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, bodyParts, x, y, y_labels, timestepSize=17, datasetType="train"):

        self.datasetType = datasetType

        if self.datasetType == "train":
            X_train = LoadData.getKeypointsfromPathList(x, bodyParts, timestepSize)
            X_train = np.asarray(X_train)
            self.X_train = torch.tensor(X_train,dtype=torch.float32).to(device)
            self.y_train = torch.tensor(y,dtype=torch.int64).to(device)
            self.inputSize = X_train.shape[2]
            print("Train: ",self.X_train.shape, self.y_train.shape)
        else:
            X_test = LoadData.getKeypointsfromPathList(x, bodyParts, timestepSize)
            X_test = np.asarray(X_test)
            self.X_test = torch.tensor(X_test,dtype=torch.float32).to(device)
            self.y_test = torch.tensor(y,dtype=torch.int64).to(device)
            self.inputSize = X_test.shape[2]
            print("Test: ",self.X_test.shape, self.y_test.shape)

        self.outputSize = len(y_labels)
        #self.weight = torch.tensor(weight, dtype=torch.float32).to(device)
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

def main(
    face: bool,
    hands: bool,
    pose: bool,
    keys_input_Path: Path,
    keypoints_input_Path: Path,
    hidden_size: int,
    lrn_rate: float,
    batch_size: int,
    num_layers: int,
    dropout: float,
    timesteps: int,
    wandb: bool,
    plot: bool,):

    ##################################################
    # 0. get started

    bodyParts = []
    if(pose):
        bodyParts = bodyParts + ["pose"]
    if(hands):
        bodyParts = bodyParts + ["hands"]
    if(face):
        bodyParts = bodyParts + ["face"]

    ##################################################
    # 1. create Dataset and DataLoader objects

    # Dataset variables
    split = 0.8
    data = pd.read_pickle(keys_input_Path+"merged.pkl")
    data = data.T

    meaning = pd.read_json(keys_input_Path+"meaning.json")

    x_path = list(data["paths"])
    y = list(data["labels"])
    y_labels = list(data["words"])
    x_timeSteps = list(data["timestepsLen"])

    X_train, X_test, y_train, y_test = train_test_split(x_path, y, train_size=split , random_state=42, stratify=y)

    dataTrainXY = SignLanguageDataset(bodyParts, X_train, y_train, meaning, timestepSize=timesteps, datasetType="train")
    dataTestXY = SignLanguageDataset(bodyParts, X_test, y_test, meaning, timestepSize=timesteps,datasetType="test")

    print("Begin predict sign language")
    np.random.seed(1)
    torch.manual_seed(1)

    # variables
    dropout = dropout
    num_layers = num_layers
    num_classes = dataTrainXY.outputSize
    batch_size = batch_size
    nEpoch = 20000
    lrn_rate = lrn_rate
    weight_decay = 0.00001
    epsilon = 1e-8
    hidden_size = hidden_size

    if wandb:
        wandbF.initConfigWandb(num_layers, num_classes, batch_size, nEpoch,
                               lrn_rate, hidden_size, dropout, weight_decay, epsilon)


    print("data train split at: %2.2f" % split)
    print("hidden size: %d" % hidden_size)
    print("dropout: %3.2f" % dropout)
    print("batch_size: %d" % batch_size)
    print("number of epoch: %d" % nEpoch)
    print("learning rate: %f" % lrn_rate)
    print("Number of layers: %d" % num_layers)

    dataTrain = torch.utils.data.DataLoader(dataTrainXY, batch_size=batch_size)
    dataTest = torch.utils.data.DataLoader(dataTestXY, batch_size=batch_size)

    ##################################################
    # 2. create neural network
    net = rnn.Net(dataTrainXY.inputSize, hidden_size, num_layers, num_classes, dropout, device).to(device)

    print('The number of parameter is: %d' % count_parameters(net))

    # Wandb the network weight
    if wandb:
        wandbF.watch(net)

    ##################################################
    # 3. train network
    net.train()  # set mode

    # loss_func = torch.nn.CrossEntropyLoss(weight=dataXY.weight)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lrn_rate,
                                 weight_decay=weight_decay, eps=epsilon)

    # In case it is necesary to recover part of the trained model from checkpoint
    #chkPntPath = ""
    # bckmod.loadFromCheckPoint(chkPntPath, net, optimizer, nEpoch)

    net.zero_grad()

    accEpochAcum = []
    lossEpochAcum = []
    accTestEpochAcum = []
    lossTestEpochAcum = []

    if plot:
        fig, axs = pp.interactivePlotConf()

    start_time = time.time()
    start_bach_time = time.time()

    for epoch in range(0, nEpoch):
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

        # if you need to save checkpoint of the model
        # chkPntPath=""
        #bckmod.saveCheckPoint(chkPntPath, net, optimizer, nEpoch)

        train_loss, train_acc = compute_loss_and_acc(loss_func, net, dataTrain)
        test_loss, test_acc = compute_loss_and_acc(loss_func, net, dataTest)

        lossEpochAcum.append(train_loss)
        accEpochAcum.append(train_acc)

        lossTestEpochAcum.append(test_loss)
        accTestEpochAcum.append(test_acc)

        if(epoch % 1 == 0):

            #Log in wandb
            if wandb:
                wandbF.wandbLog(train_loss, train_acc,
                                test_loss, test_acc)
        if(epoch % 10 == 0):
            # print epoch evaluation
            pp.printEpochEval(epoch, train_loss, train_acc, test_loss,
                              test_acc, start_bach_time)
            if plot:
                pp.plotEpochEval(fig, plt, axs, epoch, lossEpochAcum, lossTestEpochAcum,
                                accEpochAcum, accTestEpochAcum, num_layers, num_classes,
                                batch_size, nEpoch, lrn_rate, hidden_size)

            start_bach_time = time.time()

    print("Done ")
    print("Total time: %0.4f seconds" % (time.time() - start_time))
    ########################
    # END of the training section
    ##################################################
    
    # Prepare folders
    uv.createFolder("./evaluation")
    uv.createFolder("./evaluation/merged/")
    #uv.createFolder("./evaluation/rnn/classes_%d" % num_classes)
    #uv.createFolder("./evaluation/rnn/classes_%d/layers_%d" % (num_classes, num_layers))
    #uv.createFolder("./evaluation/rnn/classes_%d/layers_%d/lrnRt_%f" % (num_classes, num_layers, lrn_rate))
    #uv.createFolder("./evaluation/rnn/classes_%d/layers_%d/lrnRt_%f/batch-%d" % (num_classes, num_layers, lrn_rate, batch_size))
    pltSavePath = "./evaluation/merged"
    plt.savefig(pltSavePath + '/rnn-LOSS_lrnRt-%f_batch-%d_nEpoch-%d_hidden-%d.png' % (lrn_rate, batch_size, nEpoch, hidden_size))
    plt.savefig(pltSavePath + '/rnn-LOSS_lrnRt-%f_batch-%d_nEpoch-%d_hidden-%d.png' % (lrn_rate, batch_size, nEpoch, hidden_size))
    ##################################################
    # 4. evaluate model

    # net = Net().to(device)
    # path = ".\\trainedModels\\20WordsStateDictModel.pth"
    # net.load_state_dict(torch.load(path))

    net.eval()

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

    targetTest = torch.tensor([], dtype=torch.int64).to(device)
    output_acum = torch.tensor([], dtype=torch.float32).to(device)

    for (batch_idx, batch) in enumerate(dataTest):

        # Get data train batch
        X_kp = batch['predictors']  # inputs
        Y = batch['targets']
        Y = Y.to(device, dtype=torch.int64)

        with torch.no_grad():
            output, _ = net(X_kp)

        targetTest = torch.cat((targetTest, Y))
        output_acum = torch.cat((output_acum, output))
        
    predsTest = torch.argmax(output_acum, dim=1)

    for t, p in zip(targetTest.view(-1), predsTest.view(-1)):
        confusion_matrix_test[t.long(), p.long()] += 1

    targetTrain = torch.tensor([], dtype=torch.int64).to(device)
    output_acum = torch.tensor([], dtype=torch.float32).to(device)

    for (batch_idx, batch) in enumerate(dataTrain):

        # Get data train batch
        X_kp = batch['predictors']  # inputs
        Y = batch['targets']
        Y = Y.to(device, dtype=torch.int64)

        with torch.no_grad():
            output, _ = net(X_kp)

        targetTrain = torch.cat((targetTrain, Y))
        output_acum = torch.cat((output_acum, output))
    
    predsTrain = torch.argmax(output_acum, dim=1)

    for t, p in zip(targetTrain.view(-1), predsTrain.view(-1)):
        confusion_matrix_train[t.long(), p.long()] += 1

    print(confusion_matrix_test)
    meaning = dict(meaning[0])
    meaning = [key for (key, value) in meaning.items()]
    # print(confusion_matrix.diag()/confusion_matrix.sum(1))

    ###
    # Plot CM Test ###

    confusion_matrix_test = confusion_matrix_test.to("cpu").numpy()
    if plot:
        pp.plotConfusionMatrixTest(plt, dataTestXY, pltSavePath, confusion_matrix_test,
                                    num_layers, num_classes, batch_size, nEpoch,
                                    lrn_rate, hidden_size)

    ###
    # Plot CM Train ###

    confusion_matrix_train = confusion_matrix_train.to("cpu").numpy()

    if plot:
        pp.plotConfusionMatrixTrain(plt, dataTrainXY, pltSavePath, confusion_matrix_train,
                                    num_layers, num_classes, batch_size, nEpoch,
                                    lrn_rate, hidden_size)

    # Send confusion matrix Test to Wandb
    if wandb:
        wandbF.sendConfusionMatrix(targetTest.to("cpu").numpy(),
                                   predsTest.to("cpu").numpy(),
                                   meaning,
                                   cmTrain=False)

    # Send confusion matrix Train to Wandb
    if wandb:
        wandbF.sendConfusionMatrix(targetTrain.to("cpu").numpy(),
                                   predsTrain.to("cpu").numpy(),
                                   meaning,
                                   cmTrain=True)
    
    ##################################################
    # 5. save model

    #bckmod.saveModel(net)
    '''
    ##################################################
    # 6. make a prediction

    model = rnn.Net(dataTrainXY.inputSize, hidden_size,
                num_layers, dataTrainXY.outputSize, dropout).to(device)
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    model.load_state_dict(torch.load(path))
    '''
    if wandb:
        wandbF.finishWandb()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classification')

    parser.add_argument('--face', action="store_true",
                        help='Use holistic model: face')

    parser.add_argument('--hands', action="store_true",
                        help='Use holistic model: hands')

    parser.add_argument('--pose', action="store_true",
                        help='Use holistic model: pose')

    parser.add_argument('--keys_input_Path', type=str,
                        default="./Data/merged/AEC-PUCP_PSL_DGI156/",
                        help='relative path of keypoints input.'
                        ' Default: ./Data/AEC/Selected/')

    parser.add_argument('--keypoints_input_Path', type=str,
                        default="./Data/AEC/Keypoints/pkl/",
                        help='relative path of keypoints input.' +
                        ' Default: ./Data/keypoints/')
    
    parser.add_argument("--hidden_size", type=int, default=2000,
                        help="hidden size")

    parser.add_argument("--lrn_rate", type=float, default= 0.0000001,
                        help="learning rate")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size")

    parser.add_argument("--num_layers", type=int, default= 2,
                        help="number of layers")
                        
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout")

    parser.add_argument("--timesteps", type=int, default=150,
                        help="Number of top words")

    parser.add_argument('--wandb', action="store_true",
                        help='To activate wandb')

    parser.add_argument('--plot', action="store_true",
                        help='To activate plot from matplotlib')

    args = parser.parse_args()

    main(**vars(args))

