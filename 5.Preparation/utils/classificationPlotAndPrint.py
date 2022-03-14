#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:39:07 2021

@author: joe
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import time

def printEpochEval(epoch, lossEpoch, accEpoch, lossTestEpoch, accTestEpoch,
                   start_bach_time):

    print("================================================")
    print("epoch = %4d   loss = %0.4f" %
          (epoch, lossEpoch))
    print("acc = %0.4f" % (accEpoch))
    print("----------------------")
    print("loss (test) = %0.4f" % (lossTestEpoch))
    print("acc(test) = %0.4f" % (accTestEpoch))
    print("Epoch time: %0.4f seconds" % (time.time() - start_bach_time))

def interactivePlotConf():
    plt.ion()
    fig, axs = plt.subplots(1, 2    )
    fig.set_figheight(10)
    fig.set_figwidth(15)
    return fig, axs

def plotEpochEval(fig, plt, axs, epoch, lossEpochAcum, lossTestEpochAcum,
                  accEpochAcum, accTestEpochAcum, num_layers, num_classes, 
                  batch_size, nEpoch, lrn_rate, hidden_size):
    
    axs[0].clear()
    axs[1].clear()
    plt.title("")
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

def plotConfusionMatrixTest(plt, dataXY, pltSavePath, confusion_matrix_test, num_layers, num_classes, 
                            batch_size, nEpoch, lrn_rate, hidden_size):
    cmap=plt.cm.Blues
    normalize = False
    fig2, ax3 = plt.subplots()
    fig2.set_figheight(8)
    fig2.set_figwidth(13)
    plt.title('Num layers: %d | ' % (num_layers) +
              'batch size: %d\n' % (batch_size) +
              'num classes: %d | ' % (num_classes) +
              'nEpoch: %d\n' % (nEpoch) +
              'lrn rate: %f | ' % (lrn_rate) +
              'hidden size: %d' % (hidden_size))
    plt.imshow(confusion_matrix_test, interpolation='nearest', cmap=cmap)
    # Specify the tick marks and axis text
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, dataXY.y_labels.values(), rotation=90)
    plt.yticks(tick_marks, dataXY.y_labels.values())

    # The data formatting
    fmt = '.2f' if normalize else '.2f'
    thresh = confusion_matrix_test.max() / 2.

    # Print the text of the matrix, adjusting text colour for display
    for i, j in itertools.product(range(confusion_matrix_test.shape[0]), 
                                  range(confusion_matrix_test.shape[1])):
        plt.text(j, i, format(confusion_matrix_test[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_test[i, j] > thresh else "black")

    plt.ylabel('True label (test)')
    plt.xlabel('Predicted label (test)')
    plt.tight_layout()
    plt.savefig(pltSavePath + '/CM-TEST_lrnRt-%f_batch-%d_nEpoch-%d_hidden-%d.png' %
                (lrn_rate, batch_size, nEpoch,hidden_size))

    plt.show()

def plotConfusionMatrixTrain(plt, dataXY, pltSavePath, confusion_matrix_train,
                             num_layers, num_classes, batch_size, nEpoch,
                             lrn_rate, hidden_size):
    cmap=plt.cm.Blues
    normalize = False
    fig3, ax4 = plt.subplots()
    fig3.set_figheight(8)
    fig3.set_figwidth(13)
    plt.title('Num layers: %d | ' % (num_layers) +
              'batch size: %d\n' % (batch_size) +
              'num classes: %d | ' % (num_classes) +
              'nEpoch: %d\n' % (nEpoch) +
              'lrn rate: %f | ' % (lrn_rate) +
              'hidden size: %d' % (hidden_size))
    
    plt.imshow(confusion_matrix_train, interpolation='nearest', cmap=cmap)
    # Specify the tick marks and axis text
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, dataXY.y_labels.values(), rotation=90)
    plt.yticks(tick_marks, dataXY.y_labels.values())

    # The data formatting
    fmt = '.2f' if normalize else '.2f'
    thresh = confusion_matrix_train.max() / 2.

    # Print the text of the matrix, adjusting text colour for display
    for i, j in itertools.product(range(confusion_matrix_train.shape[0]), 
                                  range(confusion_matrix_train.shape[1])):
        plt.text(j, i, format(confusion_matrix_train[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_train[i, j] > thresh else "black")

    plt.ylabel('True label (train)')
    plt.xlabel('Predicted label (train)')

    plt.savefig(pltSavePath + '/CM-train_lrnRt-%f_batch-%d_nEpoch-%d_hidden-%d.png' %
                (lrn_rate, batch_size, nEpoch,hidden_size))
    plt.show()