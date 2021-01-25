# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 22:47:53 2021

@author: Joe
"""
# -*- coding: utf-8 -*-

# Standard library imports
import time

# Third party imports
import numpy as np
import torch

# Local imports
from utils import LoadData


device = torch.device("cpu")
# device = device("cuda")


class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, nTopWords=20):

        x, y = LoadData.getTopNWordData(nTopWords, src_file, minimun=True)

        self.x_data = torch.tensor(x, dtype=torch.float32).to(device)
        self.y_data = torch.tensor(y, dtype=torch.int64).to(device)

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

#def accuracy(model, ds): . . .

# ----------------------------------------------------

class Net(torch.nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(33, 66)  # 33-(66-66)-20
        self.drop1 = torch.nn.Dropout(0.50)
        self.hid2 = torch.nn.Linear(66, 66)
        self.drop2 = torch.nn.Dropout(0.25)
        self.oupt = torch.nn.Linear(66, 20)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x))
        z = self.drop1(z)
        z = torch.tanh(self.hid2(z))
        z = self.drop2(z)
        z = self.oupt(z)  # no softmax: CrossEntropyLoss() 
        return z

# ----------------------------------------------------

def main():
    # 0. get started
    print("Begin predict student major ")
    np.random.seed(1)
    torch.manual_seed(1)

    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    dataTrain = SignLanguageDataset(src, nTopWords=20)
    dataTrain = torch.utils.data.DataLoader(dataTrain, batch_size=5,
                                            shuffle=True)
    for epoch in range(2):
        print("\n\n Epoch = " + str(epoch))
        for (bat_idx, batch) in enumerate(dataTrain):
            print("------------------------------")
            X = batch['predictors']
            Y = batch['targets']
            print("bat_idx = " + str(bat_idx))
            #print(X)
            print(Y)

    net = Net().to(device)
    x = torch.tensor([[0.4388577342,0.4679881632,0.4834975302,0.4989468753,0.4242934585,0.4112422466,0.3983397186,0.5386477113,0.3971135318,0.4774102569,0.4253841341,0.6776406169,0.3195131123,0.7749131918,0.2366280407,0.567679286,0.3522899747,0.5117599964,0.3938273489,0.5107024312,0.4020748734,0.5160288215,0.3986841142,0.6545001864,0.4011816978,0.6560982466,0.4006917179,0.650570631,0.4034233093,0.6586065888,0.402287513,0.6184505224,0.4175108075]],
      dtype=torch.float32).to(device)
    y = net(x)
    
    print("\ninput = ")
    print(x)
    print("output = ")
    print(y)
    
    x = torch.tensor([[0.5406395197,0.5692663193,0.5841298103,0.5989130139,0.5240303874,0.5096347928,0.495405972,0.6238916516,0.4773470759,0.5664538145,0.5125743747,0.7142701745,0.3727803528,0.8483381271,0.210821867,0.735016048,0.3131961226,0.7137451768,0.3328936696,0.6926658154,0.3558881581,0.688637197,0.3610518575,0.6479050517,0.4015480876,0.6533644795,0.4134740531,0.6517947912,0.4215958714,0.655839622,0.4159375131,0.6225858927,0.4458163381],
                  [0.4388577342,0.4679881632,0.4834975302,0.4989468753,0.4242934585,0.4112422466,0.3983397186,0.5386477113,0.3971135318,0.4774102569,0.4253841341,0.6776406169,0.3195131123,0.7749131918,0.2366280407,0.567679286,0.3522899747,0.5117599964,0.3938273489,0.5107024312,0.4020748734,0.5160288215,0.3986841142,0.6545001864,0.4011816978,0.6560982466,0.4006917179,0.650570631,0.4034233093,0.6586065888,0.402287513,0.6184505224,0.4175108075]],
          dtype=torch.float32).to(device)
    y = net(x)
    
    print("\ninput = ")
    print(x)
    print("output = ")
    print(y)
    
    print("\nEnd test ")
  # 1. create Dataset and DataLoader objects
  # 2. create neural network
  # 3. train network
  # 4. evaluate model
  # 5. save model
  # 6. make a prediction 
    print("End predict student major demo ")

if __name__== "__main__":
  main()
