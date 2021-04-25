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
import matplotlib.pyplot as plt

# Local imports
from utils import LoadData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------
# 1. create Dataset and DataLoader objects
class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, nTopWords=20,
                 split=0.8, minimun=False):

        x, y, y_meaning = LoadData.getData()

        x_train, y_train, x_test, y_test = LoadData.splitData(x, y, split)

        self.inputSize = len(x[0][0])
        self.outputSize = nTopWords

        self.x_data_Test = torch.tensor(x_test, dtype=torch.float32).to(device)
        self.y_data_Test = torch.tensor(y_test, dtype=torch.int64).to(device)

        self.x_data = torch.tensor(x_train, dtype=torch.float32).to(device)
        self.y_data = torch.tensor(y_train, dtype=torch.int64).to(device)

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

    def __init__(self, inputSize, hiddenSize, numLayers, outputSize):

        super(Net, self).__init__()

        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.rnn = torch.nn.RNN(inputSize, hiddenSize, numLayers,
                                nonlinearity='relu', batch_first=True)
        self.fc = torch.nn.Linear(hiddenSize, outputSize)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        '''
        combined = torch.cat((x.unsqueeze(0), hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

        '''

        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize
                         ).to(device=device)

        out, hidden = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        out = self.softmax(out)

        return out, hidden

    # def initHidden(self):
    #    return torch.zeros(1, self.hidden_size)


# ----------------------------------------------------
# 4. evaluate model
def accuracy_eval(model, dataset):

    # assumes model.eval()
    # granular but slow approach
    n_correct = 0
    n_wrong = 0

    for i in range(len(dataset)):

        X = dataset[i]['predictors']
        Y = dataset[i]['targets']  # [0], [1], ..., [Nwords]

        with torch.no_grad():
            oupt, _ = model(X.unsqueeze(0))  # logits form

        y_pred_tags = torch.argmax(oupt)  # [0], [1], ..., [Nwords]

        probs = torch.softmax(oupt, dim=-1)  # tensor
        probs = probs.numpy()  # numpy vector prints better
        np.set_printoptions(precision=4, suppress=True)
        # print(probs)

        if y_pred_tags == Y:
            n_correct += 1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc


def accuracy_quick(yPred, yTarget):
    # assumes model.eval()
    # en masse but quick
    n = len(yTarget)

    arg_maxs = torch.argmax(yPred, dim=1)  # collapse cols

    num_correct = torch.sum(yTarget == arg_maxs)
    acc = (num_correct * 1.0 / n)
    return acc.item()


def main():

    ##################################################
    # 0. get started
    print("Begin predict sign language")
    np.random.seed(1)
    torch.manual_seed(1)

    # variables

    minimun = True
    split = 0.8

    num_layers = 5
    num_classes = 20
    batch_size = 64
    nEpoch = 5000
    lrn_rate = 0.00003
    hidden_size = 256
    sequence_length = 40

    print("minimun sizes of data: %s" % minimun)
    print("data train split at: %2.2f" % split)
    print("hidden size: %d" % hidden_size)
    print("batch_size: %d" % batch_size)
    print("number of epoch: %d", nEpoch)
    print("learning rate: %f", lrn_rate)

    ##################################################
    # 1. create Dataset and DataLoader objects

    # with open(args.output_Path+'3D/X.data','rb') as f: new_data = pkl.load(f)

    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    dataXY = SignLanguageDataset(src, nTopWords=num_classes, minimun=minimun)
    dataTrain = torch.utils.data.DataLoader(dataXY, batch_size=batch_size)

    ##################################################
    # 2. create neural network
    net = Net(dataXY.inputSize, hidden_size,
              num_layers, dataXY.outputSize).to(device)

    # In case it is necesary to recover part of the trained model
    '''
    fn = ".\\Log\\2021_01_25-10_32_57-900_checkpoint.pt"
    chkpt = torch.load(fn)
    net.load_state_dict(chkpt['net_state'])
    optimizer.load_state_dict(chkpt['optimizer_state'])
    ....
    # add thispart in netTrain
    epoch_saved = chkpt['epoch'] + 1
    for epoch in range(epoch_saved, max_epochs):
        torch.manual_seed(1 + epoch)
        # resume training as usual
    '''

    ##################################################
    # 3. train network
    net.train()  # set mode

    loss_func = torch.nn.NLLLoss()
    # loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lrn_rate)

    net.zero_grad()

    accEpochAcum = []
    lossEpochAcum = []
    accTestEpochAcum = []
    lossTestEpochAcum = []

    '''
    plt.ion()
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    '''
    for epoch in range(0, nEpoch):
        # T.manual_seed(1 + epoch)  # recovery reproducibility

        epoch_loss = 0.0  # sum avg loss per item
        epoch_acc = 0.0

        epoch_loss_test = 0.0
        epoch_acc_test = 0.0

        for (batch_idx, batch) in enumerate(dataTrain):

            # Get data from batch
            X = batch['predictors']  # inputs
            Y = batch['targets']

            X = X.to(device=device)
            Y = Y.to(device=device)

            net.train()

            optimizer.zero_grad()

            output, hidden = net(X)

            loss_val = loss_func(output, Y)
            epoch_loss += loss_val.item()  # a sum of averages
            train_acc = accuracy_quick(output, Y)
            epoch_acc += train_acc

            # Get data from test
            xTest = dataXY.x_data_Test.to(device=device)
            yTest = dataXY.y_data_Test.to(device=device)

            net.eval()
            # Test evaluation
            with torch.no_grad():
                ouptTest, _ = net(xTest)

            loss_val_test = loss_func(ouptTest, yTest)
            epoch_loss_test += loss_val_test.item()

            test_acc = accuracy_quick(ouptTest, yTest)
            epoch_acc_test += test_acc

            # Backward
            loss_val.backward()

            for p in net.parameters():
                p.data.add_(p.grad.data, alpha=-lrn_rate)

            # Step
            optimizer.step()
        '''
        dt = time.strftime("%Y_%m_%d-%H_%M_%S")
        fn = ".\\Logs\\" + str(dt) + str("-") + \
            str(epoch) + "_checkpoint.pt"

        info_dict = {
            'epoch': epoch,
            'net_state': net.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        torch.save(info_dict, fn)
        '''
        lossEpochAcum.append(epoch_loss/len(dataTrain))
        accEpochAcum.append(epoch_acc / len(dataTrain))

        lossTestEpochAcum.append(epoch_loss_test/len(dataTrain))
        accTestEpochAcum.append(epoch_acc_test / len(dataTrain))

        if(epoch % 2 == 0):
            print("================================================")
            print("epoch = %4d   loss = %0.4f" %
                  (epoch, epoch_loss/len(dataTrain)))
            print("acc = %0.4f" % (epoch_acc / len(dataTrain)))
            print("----------------------")
            print("loss (test) = %0.4f" % (epoch_loss_test/len(dataTrain)))
            print("acc(test) = %0.4f" % (epoch_acc_test / len(dataTrain)))

            '''
            axs[0].clear()
            axs[1].clear()

            axs[0].plot(range(0, epoch+1), lossEpochAcum,
                        range(0, epoch+1), lossTestEpochAcum)
            axs.flat[0].set(xlabel="Epoch",ylabel="Loss",ylim = 0.0)
            axs[0].legend(["Train", "Test"])
            axs[0].set_title("Training and Test Loss")

            axs[1].plot(range(0, epoch+1), accEpochAcum,
                        range(0, epoch+1), accTestEpochAcum)
            axs.flat[1].set(xlabel="Epoch",ylabel="Accuracy",ylim = 0.0)
            axs[1].set_ylabel("Accuracy")
            axs[1].legend(["Train", "Test"])
            axs[1].set_title("Training and Test Accuracy")
            fig.canvas.draw()
            fig.canvas.flush_events()
            '''

    print("Done ")

    plt.figure(figsize=(15, 5))

    plt.subplot(132)
    plt.plot(range(0, nEpoch), lossEpochAcum,
             range(0, nEpoch), lossTestEpochAcum)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0.0)
    plt.legend(["Train", "Test"])
    plt.title("Training and Test Loss")

    plt.subplot(133)
    plt.plot(range(0, nEpoch), accEpochAcum,
             range(0, nEpoch), accTestEpochAcum)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0)
    plt.legend(["Train", "Test"])
    plt.title("Training and Test Accuracy")
    ##################################################
    # 4. evaluate model

    # net = Net().to(device)
    # path = ".\\trainedModels\\20WordsStateDictModel.pth"
    # net.load_state_dict(torch.load(path))

    net.eval()

    src = "./Data/Keypoints/pkl/Segmented_gestures/"

    # getDatatest=True is added in order to get Test
    XY_test = SignLanguageDataset(src, nTopWords=20, getDatatest=True,
                                  minimun=minimun)

    acc = accuracy_eval(net, XY_test)
    print("\nAccuracy = %0.4f" % acc)

    ##################################################
    # 5. save model

    print("Saving trained model state dict ")
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    torch.save(net.state_dict(), path)

    ##################################################
    # 6. make a prediction
    '''
    model = Net(dataXY.inputSize).to(device)
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    model.load_state_dict(torch.load(path))

    x = torch.tensor([[0.4978884757,0.5311788917,0.5501445532,0.5689732432,0.4745099545,0.4565058947,0.4387396276,0.6012605429,0.4168290794,0.5330647826,0.4647639394,0.7237036228,0.3062814772,0.8822947145,0.1770293117,0.6423509121,0.3482455909,0.5823500752,0.3711886406,0.5668312311,0.3772580624,0.5761520267,0.3868552744,0.6829273105,0.3730428517,0.6806340218,0.3862800896,0.681951642,0.392367959,0.6943390369,0.3873288035,0.6460286975,0.428016305,0.2571010292,0.2184107155,0.2160920352,0.2146893889,0.220465675,0.2213108242,0.2220317423,0.2394188493,0.2485786676,0.3060070872,0.3151553273,0.4938592017,0.4999982715,0.8096975684,0.8287862539,0.5848222971,0.5521921515,0.5296325088,0.4863117635,0.4829116464,0.4442401528,0.5157960653,0.4852140844,1.0803204775,1.0883587599,1.4811394215,1.486025691,1.8458998203,1.8450367451,1.9062720537,1.9062017202,1.979691267,1.9778318405]],
                     dtype=torch.float32)

    with torch.no_grad():
        y = model(x)

    print("Prediction is " + str(y))
    print("End predict student major demo ")
    '''


if __name__ == "__main__":
    main()
