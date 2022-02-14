
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:15:22 2021

@author: Joe
"""

# Standard library imports
import argparse
import time
import os
import sys

# Third party imports
import torch
from sklearn.model_selection  import train_test_split
from pandas import read_pickle
import matplotlib.pyplot as plt

# Local imports
sys.path.append(os.getcwd())
import utils.video as uv
import models.resnetRnn as resnetRnn
from utils import LoadData
import utils.wandbFunctions as wandbF
import utils.classificationPlotAndPrint as pp


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

    def __init__(self, src_file, x, y, y_labels, timestepSize=17, dataType="train", split=0.8):

        self.dataType = dataType
        self.y_labels = y_labels
        
        try:
            fileData = read_pickle(args.keypoints_input_Path + str(x[0]) + '.pkl')
            fileData = LoadData.keypointsFormat(fileData, bodyParts)
            fileData = LoadData.timeStepFormat(fileData, args.timesteps)

        except:
            print("There are no instances to train the model, please check the input path")

        self.gru_input_size = len(fileData[0])

        self.outputSize = len(y_labels)
        
        if self.dataType == "train":
            self.trainSize = len(y)
            self.x_train = x
            self.y_train = torch.tensor(y, dtype=torch.int64).to(device)
        else:
            self.testSize = len(y)
            self.x_test = x
            self.y_test = torch.tensor(y, dtype=torch.int64).to(device)

    def __len__(self):

        if self.dataType == "train":
            return self.trainSize
        else:
            return self.testSize

    def __getitem__(self, index):

        if self.dataType == "train":
            img = LoadData.getXInfo(args.image_input_Path, self.x_train[index])
            img = LoadData.timeStepFormat(img, args.timesteps)
            kps = LoadData.getXInfo(args.keypoints_input_Path, self.x_train[index])
            kps = LoadData.keypointsFormat(kps, bodyParts)
            kps = LoadData.timeStepFormat(kps, args.timesteps)
            trgts = self.y_train[index]
        else:
            img = LoadData.getXInfo(args.image_input_Path, self.x_test[index])
            img = LoadData.timeStepFormat(img, args.timesteps)
            kps = LoadData.getXInfo(args.keypoints_input_Path, self.x_test[index])
            kps = LoadData.keypointsFormat(kps, bodyParts)
            kps = LoadData.timeStepFormat(kps, args.timesteps)
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


def compute_loss_and_acc(loss_func, net, dataloader):
    "compute average loss over the a dataset"
    net.eval()

    y_acum = torch.tensor([], dtype=torch.int64).to(device)
    output_acum = torch.tensor([], dtype=torch.float32).to(device)

    for (batch_idx, batch) in enumerate(dataloader):

        # Get data train batch
        X_img = batch['image']  # inputs
        X_kp = batch['keypoints']  # inputs
        Y = batch['targets']

        Y = Y.to(device, dtype=torch.int64)

        with torch.no_grad():
            output = net(x_img=X_img, x_kp=X_kp)
        
        y_acum = torch.cat((y_acum, Y))
        output_acum = torch.cat((output_acum, output))
        
    lossMean = loss_func(output_acum, y_acum)
    accMean = accuracy_quick(output_acum, y_acum)

    return lossMean.to("cpu").numpy(), accMean


def main():
    
    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    
    # Dataset variables
    timestepSize = args.timesteps
    split = 0.8
    
    x, y, weight, y_labels, x_timeSteps = LoadData.getData(args.keys_input_Path)

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=split , random_state=42, stratify=y)

    data_train = SignLanguageDataset(src, X_train, y_train, y_labels, timestepSize=timestepSize, dataType="train",split=split)
    data_test = SignLanguageDataset(src, X_test, y_test, y_labels, timestepSize=timestepSize,dataType="test",split=split)


    print("Begin predict sign language")
    torch.manual_seed(1)

    # variables
    dropout = 0.0
    gruLayers = 1
    num_classes = data_train.outputSize
    batch_size = 7
    nEpoch = 2000
    lrn_rate = 0.00005
    weight_decay = 0
    epsilon = 1e-8
    hidden_size = 250

    print("data train split at: %2.2f" % split)
    print("hidden size: %d" % hidden_size)
    print("dropout: %3.2f" % dropout)
    print("batch_size: %d" % batch_size)
    print("number of epoch: %d" % nEpoch)
    print("learning rate: %f" % lrn_rate)
    print("Number of gru layers: %d" % gruLayers)
    print("Epsilon: %3.2f" % epsilon)
    print("weight decay: %3.2f" % weight_decay)
    print("# classes: %d" % num_classes)
    
    dataTrain = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
    dataTest = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

    net = resnetRnn.resnet_rnn(num_classes, data_train.gru_input_size, hidden_size, gruLayers, dropout).to(device)
    
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

    if args.plot:
        fig, axs = pp.interactivePlotConf()

    start_time = time.time()
    
    for epoch in range(0, nEpoch):
        # T.manual_seed(1 + epoch)  # recovery reproducibility

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
            YTrain = Y.to(device, dtype=torch.int64)

            optimizer.zero_grad()

            output = net(x_img=X_img, x_kp=X_kp)

            loss_val = loss_func(output, YTrain)

            # Backward
            loss_val.backward()

            # Step
            optimizer.step()

        train_loss, train_acc = compute_loss_and_acc(loss_func, net, dataTrain)
        test_loss, test_acc = compute_loss_and_acc(loss_func, net, dataTest)


        # if you need to save checkpoint of the model
        # chkPntPath=""
        # bckmod.saveCheckPoint(chkPntPath, net, optimizer, nEpoch)


        lossEpochAcum.append(train_loss)
        accEpochAcum.append(train_acc)

        lossTestEpochAcum.append(test_loss)
        accTestEpochAcum.append(test_acc)

        if(epoch % 1 == 0):

            # print epoch evaluation
            pp.printEpochEval(epoch, train_loss, train_acc, test_loss,
                              test_acc, start_bach_time)
        if args.plot:
            pp.plotEpochEval(fig, plt, axs, epoch, lossEpochAcum, lossTestEpochAcum,
                             accEpochAcum, accTestEpochAcum, gruLayers, num_classes,
                             batch_size, nEpoch, lrn_rate, hidden_size)

    print("Done ")
    print("Total time: %0.4f seconds" % (time.time() - start_time))

    
    # Prepare folders
    uv.createFolder("./evaluation")
    uv.createFolder("./evaluation/resnet-rnn/")
    uv.createFolder("./evaluation/resnet-rnn/classes_%d" % num_classes)
    uv.createFolder("./evaluation/resnet-rnn/classes_%d/layers_%d" % (num_classes, gruLayers))
    uv.createFolder("./evaluation/resnet-rnn/classes_%d/layers_%d/lrnRt_%f" % (num_classes, gruLayers, lrn_rate))
    uv.createFolder("./evaluation/resnet-rnn/classes_%d/layers_%d/lrnRt_%f/batch-%d" % (num_classes, gruLayers, lrn_rate, batch_size))
    pltSavePath = "./evaluation/resnet-rnn/classes_%d/layers_%d/lrnRt_%f/batch-%d" % (num_classes, gruLayers, lrn_rate, batch_size)
    plt.savefig(pltSavePath + '/resnetRnn-LOSS_lrnRt-%f_batch-%d_nEpoch-%d_hidden-%d.png' % (lrn_rate, batch_size, nEpoch, hidden_size))
    
    ##################################################
    # 4. evaluate model
    
    # net = Net().to(device)
    # path = ".\\trainedModels\\20WordsStateDictModel.pth"
    # net.load_state_dict(torch.load(path))
    
    net.eval()   
    
    ########################
    # Confusion matrix (CM) TEST ###
    
    confusion_matrix_test = torch.zeros(num_classes, num_classes)
    confusion_matrix_train = torch.zeros(num_classes, num_classes)
    
    target_acum = torch.tensor([], dtype=torch.float32).to(device)
    output_acum = torch.tensor([], dtype=torch.float32).to(device)

    for (batch_idx, batch) in enumerate(dataTest):

        # Get data train batch
        X_img = batch['image']  # inputs
        X_kp = batch['keypoints']  # inputs
        Y = batch['targets']

        Y = Y.to(device, dtype=torch.int64)

        with torch.no_grad():
            output = net(x_img=X_img, x_kp=X_kp)

        target_acum = torch.cat((target_acum, Y))
        output_acum = torch.cat((output_acum, output))
        
    acc = accuracy_quick(output_acum, target_acum)
    print("=======================================")
    print("\nTest Accuracy = %0.4f" % acc)
    
    _, predsTest = torch.max(output_acum, 1)
    
    for t, p in zip(target_acum.view(-1), predsTest.view(-1)):
        confusion_matrix_test[t.long(), p.long()] += 1
        
    ###
    # Plot CM Test ###
    
    confusion_matrix_test = confusion_matrix_test.to("cpu").numpy()
    if args.plot:
        pp.plotConfusionMatrixTest(plt, data_test, pltSavePath, confusion_matrix_test,
                                   gruLayers, num_classes, batch_size, nEpoch,
                                   lrn_rate, hidden_size)
    # Send confusion matrix Test to Wandb
    if args.wandb:
        wandbF.sendConfusionMatrix(target_acum.to("cpu").numpy(),
                                   predsTest.to("cpu").numpy(),
                                   list(dataTrain.y_labels.values()),
                                   cmTrain=False)
    

    ########################
    # Confusion matrix (CM) TRAIN ###

    target_acum = torch.tensor([], dtype=torch.float32).to(device)
    output_acum = torch.tensor([], dtype=torch.float32).to(device)

    for (batch_idx, batch) in enumerate(dataTrain):

        # Get data train batch
        X_img = batch['image']  # inputs
        X_kp = batch['keypoints']  # inputs
        Y = batch['targets']

        Y = Y.to(device, dtype=torch.int64)

        with torch.no_grad():
            output = net(x_img=X_img, x_kp=X_kp)

        target_acum = torch.cat((target_acum, Y))
        output_acum = torch.cat((output_acum, output))    

    _, predsTrain = torch.max(output_acum, 1)

    for t, p in zip(target_acum.view(-1), predsTrain.view(-1)):
        confusion_matrix_train[t.long(), p.long()] += 1
    
    # print(confusion_matrix)
    # print(confusion_matrix.diag()/confusion_matrix.sum(1))
    
    ###
    # Plot CM Train ###
    
    confusion_matrix_train = confusion_matrix_train.to("cpu").numpy()
    if args.plot:
        pp.plotConfusionMatrixTrain(plt, data_train, pltSavePath, confusion_matrix_train,
                                    gruLayers, num_classes, batch_size, nEpoch,
                                    lrn_rate, hidden_size)
        
    # Send confusion matrix Train to Wandb
    if args.wandb:
        wandbF.sendConfusionMatrix(target_acum.to("cpu").numpy(),
                                   predsTrain.to("cpu").numpy(),
                                   list(dataTrain.y_labels.values()),
                                   cmTrain=True)
    '''
    ##################################################
    # 5. save model
    
    bckmod.saveModel(net)
    
    ##################################################
    # 6. make a prediction
    '''
    model = net.Net(data_train.inputSize, hidden_size,
                gruLayers, data_train.outputSize, dropout).to(device)
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
