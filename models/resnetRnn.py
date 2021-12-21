# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 21:26:46 2021

@author: Joe
"""
# Standard library imports


# Third party imports
import torch


# Local imports
from models.gru import Net as gru
from models.resnet import ResNet as resnet
from models.resnet import BasicBlock as basicBlock
from torchvision.models import resnet18

RENET_OUTPUT_SIZE = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class resnet_rnn(torch.nn.Module):

    def __init__(self,num_classes, inputSize, hiddenSize, rnnLayers, dropout):

        super(resnet_rnn, self).__init__()   

        # Resnet
        model = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(model.children())[:-2])
        for layer_index in range(4):
            for param in self.resnet[layer_index].parameters(True):
                param.requires_grad_(False)
        if embed_size != 512:
            self.pointwise_conv = nn.Conv2d(512, embed_size, 1)
        else:
            self.pointwise_conv = nn.Identity()

        #self.resnet = resnet(basicBlock, [2, 2, 2, 2], num_classes=num_classes, with_fc=False)
        self.gru = gru(inputSize+RENET_OUTPUT_SIZE, hiddenSize, rnnLayers, outputSize=num_classes, dropout=dropout)
        self.fc = torch.nn.Linear(RENET_OUTPUT_SIZE + hiddenSize, num_classes)

    def forward(self, x_img, x_kp):
        
        resnetOut = []
        for img in x_img:
            img_out = self.resnet(img)
            img_out = self.pointwise_conv(img_out)
            resnetOut.append(img_out)
        
        x_resnet = torch.stack(resnetOut)
        x = torch.cat((x_kp, x_resnet),dim=-1)

        out_2, _ = self.gru(x)


        return out_2


class dataset(torch.utils.data.Dataset):
    def __init__(self, nSamples):

        x_img = torch.randn(nSamples, 17, 3, 220, 220).to(device)
        x_kp = torch.randn(nSamples, 17, 1086).to(device)

        y = torch.randint(0, 2, (nSamples,)).to(device)

        self.inputSize = len(x_kp[0][0])
        """
        self.x_img = torch.tensor(x_img, dtype=torch.float32).to("cpu")
        self.x_kp = torch.tensor(x_kp, dtype=torch.float32).to("cpu")

        self.y_data = torch.tensor(y, dtype=torch.int64).to("cpu")
        """
        self.x_img = x_img.clone().detach().to(device)
        self.x_kp = x_kp.clone().detach().to(device)

        self.y_data = y.clone().detach().to(device)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        image = self.x_img[index].to(device)
        keypoint = self.x_kp[index].to(device)
        trgts = self.y_data[index].to(device)

        sample = {
            'image': image,
            'keypoint': keypoint,
            'targets': trgts}
        return sample


def test():

    nEpoch = 4
    lr_rate = 0.001
    weight_decay = 0.0
    epsilon = 0e-8
    batch_size = 6
    hiddenSize = 64
    rnnLayers = 1
    nSamples = 20
    num_classes = 2
    dropout = 0.0

    dataXY = dataset(nSamples)
    dataTrain = torch.utils.data.DataLoader(dataXY, batch_size=batch_size)

    net = resnet_rnn(num_classes, dataXY.inputSize, hiddenSize, rnnLayers, dropout)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_rate,
                                 weight_decay=weight_decay, eps=epsilon)
    net.zero_grad()
    for epoch in range(0, nEpoch):

        for (batch_idx, batch) in enumerate(dataTrain):
            # Get data train batch
            x_img = batch['image']  # inputs
            x_kp = batch['keypoint']  # inputs
            y = batch['targets']
            
            x_img = x_img.to(device)
            x_kp = x_kp.to(device)
            y = y.to(device, dtype=torch.int64)

            net.train().to(device)
            optimizer.zero_grad()

            output = net(x_img, x_kp).to(device)
            print(y.shape)
            loss_val = loss_func(output, y)

            # Backward
            loss_val.backward()

            # Step
            optimizer.step()
