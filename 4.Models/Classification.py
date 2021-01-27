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

# Local imports
from utils import LoadData

# from utils import testNeuralNetworkClassifier as testNNC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------
# 1. create Dataset and DataLoader objects
class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, nTopWords=20, getDatatest=False,
                 split=0.8, minimun=False):

        x, y = LoadData.getTopNWordData(nTopWords, src_file, minimun)
        x_train, y_train, x_test, y_test = LoadData.splitData(x, y, split)

        if(getDatatest):
            self.x_data = torch.tensor(x_test, dtype=torch.float32).to(device)
            self.y_data = torch.tensor(y_test, dtype=torch.int64).to(device)
        else:
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

    def __init__(self):

        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(66, 132)  # 66-(132-132)-20
        self.drop1 = torch.nn.Dropout(0.50)
        self.hid2 = torch.nn.Linear(132, 132)
        self.drop2 = torch.nn.Dropout(0.25)
        self.oupt = torch.nn.Linear(132, 20)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = self.drop1(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = self.oupt(z)  # no softmax: CrossEntropyLoss()
        return z


# ----------------------------------------------------
# 3. train network
def netTrain(net, dataTrain, parameters):

    
    return net


# accuracy (used in batch)
def multi_acc(y_pred, y_test):

    n_correct = 0
    n_wrong = 0

    for i in range(len(y_pred)):

        y_pred_tags = torch.argmax(y_pred[i])

        if y_pred_tags == y_test[i]:
            n_correct += 1
        else:
            n_wrong += 1
    acc = (n_correct * 1.0) / (n_correct + n_wrong)

    return acc


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
            oupt = model(X)  # logits form

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


"""
def accuracy_quick(model, dataset):
    # assumes model.eval()
    # en masse but quick
    n = len(dataset)
    X = dataset[0:n]['predictors']  # all X
    Y = torch.flatten(dataset[0:n]['targets'])  # 1-D

    with torch.no_grad():
        oupt = model(X)
    arg_maxs = torch.argmax(oupt, dim=1)  # collapse cols
    num_correct = torch.sum(Y==arg_maxs)
    acc = (num_correct * 1.0 / len(dataset))
    return acc.item()
"""


def main():

    ##################################################
    # 0. get started
    print("Begin predict sign language")
    np.random.seed(1)
    torch.manual_seed(1)

    # variables

    minimun = True
    split = 0.8

    batch_size = 88
    nEpoch = 1000
    lrn_rate = 0.1

    print("minimun sizes of data: %s" % minimun)
    print("data train split at: %2.2f" % split)

    print("batch_size: %d" % batch_size)
    print("number of epoch: %d", nEpoch)
    print("learning rate: %f", lrn_rate)

    ##################################################
    # 1. create Dataset and DataLoader objects
    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    dataXY = SignLanguageDataset(src, nTopWords=20, minimun=minimun)
    dataTrain = torch.utils.data.DataLoader(dataXY, batch_size=batch_size,
                                            shuffle=True)

    ##################################################
    # 2. create neural network
    net = Net().to(device)

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

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lrn_rate)

    for epoch in range(0, nEpoch):
        # T.manual_seed(1 + epoch)  # recovery reproducibility
        epoch_loss = 0.0  # sum avg loss per item
        epoch_acc = 0.0

        for (batch_idx, batch) in enumerate(dataTrain):
            X = batch['predictors']  # inputs
            Y = batch['targets']

            optimizer.zero_grad()
            oupt = net(X)

            loss_val = loss_func(oupt, Y)  # avg loss in batch
            epoch_loss += loss_val.item()  # a sum of averages

            train_acc = multi_acc(oupt, Y)
            epoch_acc += train_acc

            loss_val.backward()
            optimizer.step()

        if epoch % 100 == 0 or epoch == nEpoch-1:
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            fn = ".\\Logs\\" + str(dt) + str("-") + \
                str(epoch) + "_checkpoint.pt"

            info_dict = {
                'epoch': epoch,
                'net_state': net.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            torch.save(info_dict, fn)
            print("================================================")
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
            print("acc = %0.4f" % (epoch_acc / len(dataTrain)))

    print("Done ")

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

    model = Net().to(device)
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    model.load_state_dict(torch.load(path))

    x = torch.tensor([[0.4978884757,0.5311788917,0.5501445532,0.5689732432,0.4745099545,0.4565058947,0.4387396276,0.6012605429,0.4168290794,0.5330647826,0.4647639394,0.7237036228,0.3062814772,0.8822947145,0.1770293117,0.6423509121,0.3482455909,0.5823500752,0.3711886406,0.5668312311,0.3772580624,0.5761520267,0.3868552744,0.6829273105,0.3730428517,0.6806340218,0.3862800896,0.681951642,0.392367959,0.6943390369,0.3873288035,0.6460286975,0.428016305,0.2571010292,0.2184107155,0.2160920352,0.2146893889,0.220465675,0.2213108242,0.2220317423,0.2394188493,0.2485786676,0.3060070872,0.3151553273,0.4938592017,0.4999982715,0.8096975684,0.8287862539,0.5848222971,0.5521921515,0.5296325088,0.4863117635,0.4829116464,0.4442401528,0.5157960653,0.4852140844,1.0803204775,1.0883587599,1.4811394215,1.486025691,1.8458998203,1.8450367451,1.9062720537,1.9062017202,1.979691267,1.9778318405]],
                     dtype=torch.float32)

    with torch.no_grad():
        y = model(x)

    print("Prediction is " + str(y))
    print("End predict student major demo ")


if __name__ == "__main__":
    main()
