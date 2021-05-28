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
import itertools

# Third party imports
import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports
from utils import LoadData

torch.cuda.empty_cache()
device = torch.device("cuda" )
print("############ ", device, " ############")

# ----------------------------------------------------
# 1. create Dataset and DataLoader objects
class SignLanguageDataset(torch.utils.data.Dataset):

    def __init__(self, src_file, nTopWords=10,
                 split=0.8):

        x, y, weight, y_labels = LoadData.getData()

        x_train, y_train, x_test, y_test = LoadData.splitData(x, y, split,
                                                              leastValue=True,
                                                              balancedTest=True,
                                                              doShuffle=False)

        self.inputSize = len(x[0][0])
        self.outputSize = nTopWords

        self.weight = torch.tensor(weight, dtype=torch.float32).to(device)
        self.y_meaning = y_labels

        self.x_data_Test = torch.tensor(x_test, dtype=torch.float32)
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
                                 batch_first=True)

        self.fc = torch.nn.Linear(hiddenSize, outputSize)

    def forward(self, x):

        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize
                         ).to(device=device)

        out, hidden = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out, hidden


# ----------------------------------------------------
# 4. evaluate model
def accuracy_eval(model, x, y):

    # assumes model.eval()
    # granular but slow approach
    n_correct = 0
    n_wrong = 0

    for i in range(len(y)):

        X = x
        Y = y  # [0], [1], ..., [Nwords]

        with torch.no_grad():
            oupt, _ = model(X)  # logits form

        y_pred_tags = torch.argmax(oupt, dim=1)  # [0], [1], ..., [Nwords]

        print(y_pred_tags.shape, Y.shape, oupt.shape)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    ##################################################
    # 0. get started
    print("Begin predict sign language")
    np.random.seed(1)
    torch.manual_seed(1)

    # variables

    minimun = True
    split = 0.8

    num_layers = 100
    num_classes = 10
    batch_size = 30
    nEpoch = 500
    lrn_rate = 0.001
    hidden_size = 512
    # sequence_length = 40

    print("minimun sizes of data: %s" % minimun)
    print("data train split at: %2.2f" % split)
    print("hidden size: %d" % hidden_size)
    print("batch_size: %d" % batch_size)
    print("number of epoch: %d" % nEpoch)
    print("learning rate: %f" % lrn_rate)
    print("Number of layers: %d" % num_layers)

    ##################################################
    # 1. create Dataset and DataLoader objects

    # with open(args.output_Path+'3D/X.data','rb') as f: new_data = pkl.load(f)

    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    dataXY = SignLanguageDataset(src, nTopWords=num_classes)
    dataTrain = torch.utils.data.DataLoader(dataXY, batch_size=batch_size)

    ##################################################
    # 2. create neural network
    net = Net(dataXY.inputSize, hidden_size,
              num_layers, dataXY.outputSize).to(device)
    print('The number of parameter is: %d' % count_parameters(net))
    # print(pytorch_total_params)
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

    # loss_func = torch.nn.NLLLoss()
    # loss_func = torch.nn.CrossEntropyLoss(weight=dataXY.weight)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lrn_rate)

    net.zero_grad()

    accEpochAcum = []
    lossEpochAcum = []
    accTestEpochAcum = []
    lossTestEpochAcum = []

    plt.ion()
    fig, axs = plt.subplots(1, 2    )
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    
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
    # def initHidden(self):
    # return torch.zeros(1, self.hidden_size)

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

        if(epoch % 1 == 0):
            print("================================================")
            print("epoch = %4d   loss = %0.4f" %
                  (epoch, epoch_loss/len(dataTrain)))
            print("acc = %0.4f" % (epoch_acc / len(dataTrain)))
            print("----------------------")
            print("loss (test) = %0.4f" % (epoch_loss_test/len(dataTrain)))
            print("acc(test) = %0.4f" % (epoch_acc_test / len(dataTrain)))
            print("Epoch time: %0.4f seconds" % (time.time() - start_bach_time))
            axs[0].clear()
            axs[1].clear()
            plt.title("hola")
            axs[0].plot(range(0, epoch+1), lossEpochAcum,
                        range(0, epoch+1), lossTestEpochAcum)
            axs.flat[0].set(xlabel="Epoch",ylabel="Loss",ylim = 0.0)
            axs[0].legend(["Train", "Test"])
            axs[0].set_title("Loss (CrossEntropyLoss)")
            fig.suptitle('Num layers: %d | ' % (num_layers) +
                         'batch size: %d\n' % (batch_size) +
                         'num classes: %d | ' % (num_classes) +
                         'nEpoch: %d\n' % (nEpoch) +
                         'lrn rate: %f | ' % (lrn_rate) +
                         'hidden size: %d' % (hidden_size))

            axs[1].plot(range(0, epoch+1), accEpochAcum,
                        range(0, epoch+1), accTestEpochAcum)
            axs.flat[1].set(xlabel="Epoch",ylabel="Accuracy",ylim = 0.0)
            axs[1].set_ylabel("Accuracy")
            axs[1].legend(["Train", "Test"])
            axs[1].set_title("Accuracy")
            fig.canvas.draw()
            fig.canvas.flush_events()

    print("Done ")
    print("Total time: %0.4f seconds" % (time.time() - start_time))
    ##################################################
    # 4. evaluate model

    # net = Net().to(device)
    # path = ".\\trainedModels\\20WordsStateDictModel.pth"
    # net.load_state_dict(torch.load(path))

    net.eval()

    src = "./Data/Keypoints/pkl/Segmented_gestures/"

    X_test = dataXY.x_data_Test.to(device)
    Y_test = dataXY.y_data_Test.to(device)

    with torch.no_grad():
        ouptTest, _ = net(X_test)

    acc = accuracy_quick(ouptTest, Y_test)
    print("=======================================")
    print("\nTest Accuracy = %0.4f" % acc)

    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():

        inputs = dataXY.x_data_Test.to(device)
        classes = dataXY.y_data_Test.to(device)

        outputs, _ = net(inputs)

        _, preds = torch.max(outputs, 1)

        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)
    print(confusion_matrix.diag()/confusion_matrix.sum(1))

    confusion_matrix = confusion_matrix.to("cpu").numpy()
    classes = classes.to("cpu").numpy()

    cmap=plt.cm.Blues

    normalize = False
    fig2, ax3 = plt.subplots()
    fig2.set_figheight(13)
    fig2.set_figwidth(15)
    plt.title('Num layers: %d | ' % (num_layers) +
              'batch size: %d\n' % (batch_size) +
              'num classes: %d | ' % (num_classes) +
              'nEpoch: %d\n' % (nEpoch) +
              'lrn rate: %f | ' % (lrn_rate) +
              'hidden size: %d' % (hidden_size))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    # Specify the tick marks and axis text
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, dataXY.y_meaning.values(), rotation=90)
    plt.yticks(tick_marks, dataXY.y_meaning.values())

    # The data formatting
    fmt = '.2f' if normalize else '.2f'
    thresh = confusion_matrix.max() / 2.

    # Print the text of the matrix, adjusting text colour for display
    for i, j in itertools.product(range(confusion_matrix.shape[0]), 
                                  range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    ##################################################
    # 5. save model

    print("Saving trained model state dict ")
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    torch.save(net.state_dict(), path)

    ##################################################
    # 6. make a prediction

    model = Net(dataXY.inputSize, hidden_size,
              num_layers, dataXY.outputSize).to(device)
    path = ".\\trainedModels\\20WordsStateDictModel.pth"
    model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    main()

# 20, 50 y 100
# 0.01 y 0.001
# 10 clases
# 40 * 20 => 800 (parámetros)
# 20 * 20 => 400 *3 (cell []hid, aou, ) = 1200
# 800 + 1200 = 2000 (parametros)
# matriz de confusion
# limitar las instancias
# 