# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 22:47:53 2021

@author: Joe

The following code was created using these guides:
    (and doing some modifications)

https://visualstudiomagazine.com/articles/2020/12/04/multiclass-pytorch.aspx
https://visualstudiomagazine.com/articles/2020/12/15/pytorch-network.aspx
https://visualstudiomagazine.com/articles/2021/01/04/pytorch-training.aspx
https://visualstudiomagazine.com/articles/2021/01/25/pytorch-model-accuracy.aspx

works in windows 10 with:

pytorch: 1.7.1 (conda)
python: 3.8.6
conda: 4.9.2

"""
# -*- coding: utf-8 -*-

# Standard library imports
import argparse
import time
import os
import sys

# Third party imports
import numpy as np
import torch
import matplotlib.pyplot as plt
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("############ ", device, " ############")

parser = argparse.ArgumentParser(description='Classification')

parser.add_argument('--face', action="store_true",
                    help='Use holistic model: face')

parser.add_argument('--hands', action="store_true",
                    help='Use holistic model: hands')

parser.add_argument('--pose', action="store_true",
                    help='Use holistic model: pose')

parser.add_argument('--keys_input_Path', type=str,
                    default="./Data/AEC/Selected/",
                    help='relative path of keypoints input.'
                    ' Default: ./Data/AEC/Selected/')

parser.add_argument('--keypoints_input_Path', type=str,
                    default="./Data/AEC/Keypoints/pkl/",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/keypoints/')

parser.add_argument("--timesteps", type=int, default=17,
                    help="Number of top words")

parser.add_argument('--wandb', action="store_true",
                    help='To activate wandb')

parser.add_argument('--plot', action="store_true",
                    help='To activate plot from matplotlib')

args = parser.parse_args()

bodyParts = []
if(args.pose):
    bodyParts = bodyParts + ["pose"]
if(args.hands):
    bodyParts = bodyParts + ["hands"]
if(args.face):
    bodyParts = bodyParts + ["face"]

# ----------------------------------------------------
# 1. create Dataset and DataLoader objects


class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, x, y, y_labels, timestepSize=17, datasetType="train", split=0.8):

        self.datasetType = datasetType

        if self.datasetType == "train":
            X_train = LoadData.getKeypointsfromIdList(args.keypoints_input_Path, x, bodyParts, timestepSize)
            self.X_train = torch.tensor(X_train,dtype=torch.float32).to(device)
            self.y_train = torch.tensor(y,dtype=torch.int64).to(device)
            self.inputSize = X_train.shape[2]
            print("Train: ",self.X_train.shape, self.y_train.shape)
        else:
            X_test = LoadData.getKeypointsfromIdList(args.keypoints_input_Path, x, bodyParts, timestepSize)
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

def main():

    ##################################################
    # 0. get started

    ##################################################
    # 1. create Dataset and DataLoader objects

    # with open(args.output_Path+'3D/X.data','rb') as f: new_data = pkl.load(f)

    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    
    # Dataset variables
    timestepSize = args.timesteps
    split = 0.8
    
    x, y, weight, y_labels, x_timeSteps = LoadData.getData(args.keys_input_Path)

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=split , random_state=42, stratify=y)

    dataTrainXY = SignLanguageDataset(src, X_train, y_train, y_labels, timestepSize=timestepSize, datasetType="train",split=split)
    dataTestXY = SignLanguageDataset(src, X_test, y_test, y_labels, timestepSize=timestepSize,datasetType="test",split=split)

    print("Begin predict sign language")
    np.random.seed(1)
    torch.manual_seed(1)

    # variables
    dropout = 0.0
    num_layers = 1
    num_classes = dataTrainXY.outputSize
    batch_size = 32
    nEpoch = 2000
    lrn_rate = 0.00004
    weight_decay = 0
    epsilon = 1e-8
    hidden_size = 700

    if args.wandb:
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
    net = rnn.Net(dataTrainXY.inputSize, hidden_size, num_layers, num_classes, dropout).to(device)

    print('The number of parameter is: %d' % count_parameters(net))

    # Wandb the network weight
    if args.wandb:
        wandbF.watch(net)

    ##################################################
    # 3. train network
    net.train()  # set mode

    # loss_func = torch.nn.CrossEntropyLoss(weight=dataXY.weight)
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

    if args.plot:
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
        # bckmod.saveCheckPoint(chkPntPath, net, optimizer, nEpoch)

        train_loss, train_acc = compute_loss_and_acc(loss_func, net, dataTrain)
        test_loss, test_acc = compute_loss_and_acc(loss_func, net, dataTest)

        lossEpochAcum.append(train_loss)
        accEpochAcum.append(train_acc)

        lossTestEpochAcum.append(test_loss)
        accTestEpochAcum.append(test_acc)

        if(epoch % 1 == 0):

            #Log in wandb
            if args.wandb:
                wandbF.wandbLog(train_loss, train_acc,
                                test_loss, test_acc)
        if(epoch % 100 == 0):
            # print epoch evaluation
            pp.printEpochEval(epoch, train_loss, train_acc, test_loss,
                              test_acc, start_bach_time)
            if args.plot:
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
    uv.createFolder("./evaluation/rnn/")
    uv.createFolder("./evaluation/rnn/classes_%d" % num_classes)
    uv.createFolder("./evaluation/rnn/classes_%d/layers_%d" % (num_classes, num_layers))
    uv.createFolder("./evaluation/rnn/classes_%d/layers_%d/lrnRt_%f" % (num_classes, num_layers, lrn_rate))
    uv.createFolder("./evaluation/rnn/classes_%d/layers_%d/lrnRt_%f/batch-%d" % (num_classes, num_layers, lrn_rate, batch_size))
    pltSavePath = "./evaluation/rnn/classes_%d/layers_%d/lrnRt_%f/batch-%d" % (num_classes, num_layers, lrn_rate, batch_size)
    plt.savefig(pltSavePath + '/rnn-LOSS_lrnRt-%f_batch-%d_nEpoch-%d_hidden-%d.png' % (lrn_rate, batch_size, nEpoch, hidden_size))
    
    ##################################################
    # 4. evaluate model

    # net = Net().to(device)
    # path = ".\\trainedModels\\20WordsStateDictModel.pth"
    # net.load_state_dict(torch.load(path))

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

    # print(confusion_matrix)
    # print(confusion_matrix.diag()/confusion_matrix.sum(1))

    ###
    # Plot CM Test ###

    confusion_matrix_test = confusion_matrix_test.to("cpu").numpy()
    if args.plot:
        pp.plotConfusionMatrixTest(plt, dataTestXY, pltSavePath, confusion_matrix_test,
                                    num_layers, num_classes, batch_size, nEpoch,
                                    lrn_rate, hidden_size)

    ###
    # Plot CM Train ###

    confusion_matrix_train = confusion_matrix_train.to("cpu").numpy()

    if args.plot:
        pp.plotConfusionMatrixTrain(plt, dataTrainXY, pltSavePath, confusion_matrix_train,
                                    num_layers, num_classes, batch_size, nEpoch,
                                    lrn_rate, hidden_size)

    # Send confusion matrix Test to Wandb
    if args.wandb:
        wandbF.sendConfusionMatrix(targetTest.to("cpu").numpy(),
                                   predsTest.to("cpu").numpy(),
                                   list(dataTrain.y_labels.values()),
                                   cmTrain=False)

    # Send confusion matrix Train to Wandb
    if args.wandb:
        wandbF.sendConfusionMatrix(targetTrain.to("cpu").numpy(),
                                   predsTrain.to("cpu").numpy(),
                                   list(dataTrain.y_labels.values()),
                                   cmTrain=True)
    '''
    ##################################################
    # 5. save model

    bckmod.saveModel(net)

    ##################################################
    # 6. make a prediction

    model = rnn.Net(dataTrainXY.inputSize, hidden_size,
                num_layers, dataTrainXY.outputSize, dropout).to(device)
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    model.load_state_dict(torch.load(path))
    '''
    if args.wandb:
        wandbF.finishWandb()


if __name__ == "__main__":
    main()
    '''
    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    dataXY = SignLanguageDataset(src, nTopWords=10)
    dataTrain = torch.utils.data.DataLoader(dataXY, batch_size=8)
    for (batch_idx, batch) in enumerate(dataTrain):
        X = batch['predictors']  # inputs
        Y = batch['targets']
        XTrain = X.to(device)
        YTrain = Y.to(device)
    torch.device('cuda')
    meaning = list(dataXY.y_labels.values())

    print(meaning)
    '''
