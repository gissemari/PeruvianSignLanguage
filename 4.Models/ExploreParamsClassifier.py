# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:58:01 2021

@author: Joe

This Script is similar than Classification.py but some "def" has been created
to make Cross-validation (using ax-plataform) able to use and run


these modification has been made using this guide:
    (and doing some modifications)

https://towardsdatascience.com/quick-tutorial-using-bayesian-optimization-to-tune-your-hyperparameters-in-pytorch-e9f74fc133c2

works in windows 10 with:

ax-plataform: 0.1.19
pytorch: 1.7.1
python: 3.8.6
conda: 4.9.2

"""
# -*- coding: utf-8 -*-

# Standard library imports


# Third party imports
import numpy as np
import torch
import torch.optim as optim
import plotly.graph_objects as go
from ax.service.managed_loop import optimize
from ax.plot.trace import optimization_trace_single_method
from ax.plot.contour import plot_contour

# Local imports
from utils import LoadData

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

    def __init__(self, h1, h2):

        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(66, h1)  # 66-(132-132)-20
        self.drop1 = torch.nn.Dropout(0.50)
        self.hid2 = torch.nn.Linear(h1, h2)
        self.drop2 = torch.nn.Dropout(0.25)
        self.oupt = torch.nn.Linear(h2, 20)

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

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=parameters.get("lr", 0.1),  # 0.1 is used if no lr is specified
                                momentum=parameters.get("momentum", 0.9))

    scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=int(parameters.get("step_size", 30)),
      gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
       )

    nEpoch = parameters.get("max_epoch", 1000)

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
            scheduler.step()

        if epoch % 100 == 0 or epoch == nEpoch-1:
            print("================================================")
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
            print("acc = %0.4f" % (epoch_acc / len(dataTrain)))

    print("Done ")
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


def initNet(parameterization):
    ##################################################
    # 0. get started
    print("Begin predict sign language")
    np.random.seed(1)
    torch.manual_seed(1)

    ##################################################
    # 2. create neural network

    h1 = parameterization.get("h1", 132)
    h2 = parameterization.get("h2", 132)

    net = Net(h1, h2).to(device)

    # In case it is necesary to recover part of the trained model
    '''
    fn = ".\\Log\\2021_01_25-10_32_57-900_checkpoint.pt"
    chkpt = torch.load(fn)
    net.load_state_dict(chkpt['net_state'])
    optimizer.load_state_dict(chkpt['optimizer_state'])
    ....
    # add this part in netTrain
    epoch_saved = chkpt['epoch'] + 1
    for epoch in range(epoch_saved, max_epochs):
        torch.manual_seed(1 + epoch)
        # resume training as usual
    '''
    return net


def save_plot(plot, path):

    data = plot[0]['data']
    lay = plot[0]['layout']

    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image(path)


def trainEvaluate(parameterization):
    print(parameterization)
    # Get neural net
    untrained_net = initNet(parameterization)

    src = "./Data/Keypoints/pkl/Segmented_gestures/"
    minimun = True
    batch_size = parameterization.get("batchsize", 88)

    dataXY = SignLanguageDataset(src, nTopWords=20, minimun=minimun)
    dataTrain = torch.utils.data.DataLoader(dataXY, batch_size=batch_size,
                                            shuffle=True)

    # getDatatest=True is added in order to get Test
    XY_test = SignLanguageDataset(src, nTopWords=20, getDatatest=True,
                                  minimun=minimun)

    # train
    trained_net = netTrain(net=untrained_net, dataTrain=dataTrain,
                           parameters=parameterization)

    # return the accuracy of the model as it was trained in this run
    return accuracy_eval(
        model=trained_net,
        dataset=XY_test,
    )


def main():

    # Cross-validation
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4],
             "log_scale": True},
            {"name": "batchsize", "type": "range", "bounds": [16, 128]},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            # {"name": "h1", "type": "range", "bounds": [16, 256]},
            # {"name": "h2", "type": "range", "bounds": [16, 256]},
            # {"name": "max_epoch", "type": "range", "bounds": [1000, 10000]},
            # {"name": "stepsize", "type": "range", "bounds": [20, 40]},
            ],
        total_trials=20,  # 20 is the default
        evaluation_function=trainEvaluate,
        objective_name='accuracy'
    )

    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)

    best_objectives = np.array([[trial.objective_mean*100
                                 for trial in experiment.trials.values()]])

    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )
    save_plot(best_objective_plot, "./plot_best_objective.pdf")

    plot_lr_batch = (plot_contour(model=model, param_x='batchsize',
                                  param_y='lr', metric_name='accuracy'))
    save_plot(plot_lr_batch, "./plot_contour_lr_vs_batchsize.pdf")

    plot_lr_moment = (plot_contour(model=model, param_x='momentum',
                                   param_y='lr', metric_name='accuracy'))
    save_plot(plot_lr_moment, "./plot_contour_lr_vs_momentun.pdf")

    # uncomment if use h1 and h2 in parameters of optimize
    # plot_h1_h2 = (plot_contour(model=model, param_x='h1', param_y='h2', metric_name='accuracy'))
    # save_plot(plot_h1_h2, "./plot_contour_h1_vs_h2.pdf")


if __name__ == "__main__":
    main()
