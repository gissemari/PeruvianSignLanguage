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
import time

# Third party imports
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

# Local imports
from utils import LoadData
import utils.video as uv
import utils.wandbFunctions as wandbF
import utils.backupModel as bckmod
import utils.classificationPlotAndPrint as pp

torch.cuda.empty_cache()
device = torch.device("cpu")
print("############ ", device, " ############")

parser = argparse.ArgumentParser(description='Classification')

# 3D boolean
parser.add_argument('--wandb', action="store_true",
                    help='To activate wandb')

args = parser.parse_args()

# ----------------------------------------------------
# 1. create Dataset and DataLoader objects


class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, split=0.8):

        x, y, weight, y_labels, x_timeSteps = LoadData.getData()

        x_train, y_train, x_test, y_test = LoadData.splitData(x, y,
                                                              x_timeSteps,
                                                              split=split,
                                                              leastValue=True,
                                                              balancedTest=True,
                                                              fixed=False,
                                                              fixedTest=2,
                                                              doShuffle=True)

        self.inputSize = len(x[0][0])
        self.outputSize = len(y_labels)

        self.weight = torch.tensor(weight, dtype=torch.float32).to(device)
        self.y_labels = y_labels

        self.x_data_Test = torch.tensor(x_test, dtype=torch.float32).to(device)
        self.y_data_Test = torch.tensor(y_test, dtype=torch.int64).to(device)

        self.x_data = torch.tensor(x_train, dtype=torch.float32).to(device)
        self.y_data = torch.tensor(y_train, dtype=torch.int64).to(device)
        print(self.x_data.shape, self.y_data.shape)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        preds = self.x_data[index]
        trgts = self.y_data[index]

        sample = {
            'predictors': preds,
            'targets': trgts}
        return sample


# ----------------------------------------------------
# 2. create neural network
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

        h0 = torch.zeros(self.numLayers, x.size(
            0), self.hiddenSize).to(device=device)
        out, hidden = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out, hidden


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

    ##################################################
    # 0. get started

    ##################################################
    # 1. create Dataset and DataLoader objects

    # with open(args.output_Path+'3D/X.data','rb') as f: new_data = pkl.load(f)

    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    dataXY = SignLanguageDataset(src)

    print("Begin predict sign language")
    np.random.seed(1)
    torch.manual_seed(1)

    # variables
    minimun = True
    split = 0.8
    dropout = 0
    num_layers = 3
    num_classes = dataXY.outputSize
    batch_size = 6
    nEpoch = 2000
    lrn_rate = 0.0001
    weight_decay = 0
    epsilon = 1e-3
    hidden_size = 10
    # sequence_length = 40

    if args.wandb:
        wandbF.initConfigWandb(num_layers, num_classes, batch_size, nEpoch,
                               lrn_rate, hidden_size, dropout, weight_decay, epsilon)
     

    print("minimun sizes of data: %s" % minimun)
    print("data train split at: %2.2f" % split)
    print("hidden size: %d" % hidden_size)
    print("batch_size: %d" % batch_size)
    print("number of epoch: %d" % nEpoch)
    print("learning rate: %f" % lrn_rate)
    print("Number of layers: %d" % num_layers)

    dataTrain = torch.utils.data.DataLoader(dataXY, batch_size=batch_size)

    ##################################################
    # 2. create neural network
    net = Net(dataXY.inputSize, hidden_size,
              num_layers, dataXY.outputSize, dropout).to(device)

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

    fig, axs = pp.interactivePlotConf()

    start_time = time.time()

    for epoch in range(0, nEpoch):
        # T.manual_seed(1 + epoch)  # recovery reproducibility

        epoch_loss = 0.0  # sum avg loss per item
        epoch_acc = 0.0

        epoch_loss_test = 0.0
        epoch_acc_test = 0.0

        start_bach_time = time.time()

        for (batch_idx, batch) in enumerate(dataTrain):

            # Get data train batch
            X = batch['predictors']  # inputs
            Y = batch['targets']
            XTrain = X.to(device)
            YTrain = Y.to(device)

            # Test evaluation
            net.train()

            optimizer.zero_grad()

            output, hidden = net(XTrain)

            loss_val = loss_func(output, YTrain)
            epoch_loss += loss_val.item()  # a sum of averages
            train_acc = accuracy_quick(output, YTrain)
            epoch_acc += train_acc

            # Get data from test
            xTest = dataXY.x_data_Test.to(device)
            yTest = dataXY.y_data_Test.to(device)

            # Backward
            loss_val.backward()

            # Step
            optimizer.step()

            # Test evaluation
            net.eval()

            with torch.no_grad():
                ouptTest, _ = net(xTest)

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

            #Log in wandb
            if args.wandb:
                wandbF.wandbLog(lossEpoch, accEpoch,
                                lossTestEpoch, accTestEpoch)

            # print epoch evaluation
            pp.printEpochEval(epoch, lossEpoch, accEpoch, lossTestEpoch,
                              accTestEpoch, start_bach_time)

            pp.plotEpochEval(fig, plt, axs, epoch, lossEpochAcum, lossTestEpochAcum,
                             accEpochAcum, accTestEpochAcum, num_layers, num_classes,
                             batch_size, nEpoch, lrn_rate, hidden_size)

    print("Done ")
    print("Total time: %0.4f seconds" % (time.time() - start_time))
    ########################
    # END of the training section
    ##################################################

    # Prepare folders
    uv.createFolder("./evaluation/classes_%d" % num_classes)
    uv.createFolder("./evaluation/classes_%d/layers_%d" %
                    (num_classes, num_layers))
    uv.createFolder("./evaluation/classes_%d/layers_%d/lrnRt_%f" %
                    (num_classes, num_layers, lrn_rate))
    uv.createFolder("./evaluation/classes_%d/layers_%d/lrnRt_%f/batch-%d" %
                    (num_classes, num_layers, lrn_rate, batch_size))
    pltSavePath = "./evaluation/classes_%d/layers_%d/lrnRt_%f/batch-%d" % (
        num_classes, num_layers, lrn_rate, batch_size)
    plt.savefig(pltSavePath + '/LOSS_lrnRt-%f_batch-%d_nEpoch-%d_hidden-%d.png' %
                (lrn_rate, batch_size, nEpoch, hidden_size))

    ##################################################
    # 4. evaluate model

    # net = Net().to(device)
    # path = ".\\trainedModels\\20WordsStateDictModel.pth"
    # net.load_state_dict(torch.load(path))

    net.eval()

    src = "./Data/Keypoints/pkl/Segmented_gestures/"

    ###
    # Test Accuracy ###

    X_test = dataXY.x_data_Test.to(device)
    Y_test = dataXY.y_data_Test.to(device)

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
        inputsTest = dataXY.x_data_Test.to(device)
        targetTest = dataXY.y_data_Test.to(device)

        outputsTest, _ = net(inputsTest)

        _, predsTest = torch.max(outputsTest, 1)

        for t, p in zip(targetTest.view(-1), predsTest.view(-1)):
            confusion_matrix_test[t.long(), p.long()] += 1

        # CM Train
        inputsTrain = dataXY.x_data.to(device)
        targetTrain = dataXY.y_data.to(device)

        outputsTrain, _ = net(inputsTrain)

        _, predsTrain = torch.max(outputsTrain, 1)

        for t, p in zip(targetTrain.view(-1), predsTrain.view(-1)):
            confusion_matrix_train[t.long(), p.long()] += 1

    # print(confusion_matrix)
    # print(confusion_matrix.diag()/confusion_matrix.sum(1))

    ###
    # Plot CM Test ###

    confusion_matrix_test = confusion_matrix_test.to("cpu").numpy()

    pp.plotConfusionMatrixTest(plt, dataXY, pltSavePath, confusion_matrix_test,
                               num_layers, num_classes, batch_size, nEpoch,
                               lrn_rate, hidden_size)

    ###
    # Plot CM Train ###

    confusion_matrix_train = confusion_matrix_train.to("cpu").numpy()

    pp.plotConfusionMatrixTrain(plt, dataXY, pltSavePath, confusion_matrix_train,
                                num_layers, num_classes, batch_size, nEpoch,
                                lrn_rate, hidden_size)

    # Send confusion matrix Test to Wandb
    if args.wandb:
        wandbF.sendConfusionMatrix(targetTest.to("cpu").numpy(),
                                   predsTest.to("cpu").numpy(),
                                   list(dataXY.y_labels.values()),
                                   cmTrain=False)

    # Send confusion matrix Train to Wandb
    if args.wandb:
        wandbF.sendConfusionMatrix(targetTrain.to("cpu").numpy(),
                                   predsTrain.to("cpu").numpy(),
                                   list(dataXY.y_labels.values()),
                                   cmTrain=True)

    ##################################################
    # 5. save model

    bckmod.saveModel(net)

    ##################################################
    # 6. make a prediction

    model = Net(dataXY.inputSize, hidden_size,
                num_layers, dataXY.outputSize, dropout).to(device)
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    model.load_state_dict(torch.load(path))
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
