"""PyTorch Lightning module definition.
Delegates computation to one of the defined networks (vtn.py, vtn_hc.py, vtn_hcpf.py)"""
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR

from .vtn_hcpf import VTNHCPF
from .vtn_hcpf_d import VTNHCPFD
import torchmetrics

import numpy as np
import pandas as pd
import os.path

def get_model_def():
    return Module


def get_model(**kwargs):
    return Module(**kwargs)


class Module(pl.LightningModule):

    def __init__(self, model, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        NUM_CLASSES = self.hparams.num_classes

        print("Num Classes:", self.hparams.num_classes)

        if model == 'VTN_HCPF':
            self.model = VTNHCPF(NUM_CLASSES, self.hparams.num_heads, self.hparams.num_layers, self.hparams.embed_size,
                                 self.hparams.sequence_length, self.hparams.cnn,
                                 self.hparams.freeze_layers,
                                 self.hparams.dropout, device=self.device)
        elif model == 'VTN_HCPF_D':
            self.model = VTNHCPFD(NUM_CLASSES, self.hparams.num_heads, self.hparams.num_layers, self.hparams.embed_size,
                                  self.hparams.sequence_length, self.hparams.cnn,
                                  self.hparams.freeze_layers,
                                  self.hparams.dropout, device=self.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics have been changed following this guide
        # https://www.exxactcorp.com/blog/Deep-Learning/advanced-pytorch-lightning-using-torchmetrics-and-lightning-flash

        self.train_acc = torchmetrics.Accuracy()
        self.train_f1_micro = torchmetrics.F1(num_classes=NUM_CLASSES, average="micro")
        self.train_f1_macro = torchmetrics.F1(num_classes=NUM_CLASSES, average="macro")
        #self.train_auroc = torchmetrics.AUROC(num_classes=NUM_CLASSES, average="micro")
        self.val_acc = torchmetrics.Accuracy()
        self.val_f1_micro = torchmetrics.F1(num_classes=NUM_CLASSES, average="micro")
        self.val_f1_macro = torchmetrics.F1(num_classes=NUM_CLASSES, average="macro")
        #self.val_auroc = torchmetrics.AUROC(num_classes=NUM_CLASSES, average="micro")

        self.epochCount = 0

        self.best_val_acc = np.NINF
        self.best_val_f1_micro = np.NINF
        self.best_val_f1_macro = np.NINF
        self.best_val_loss = np.inf

        self.best_train_acc = np.NINF
        self.best_train_f1_micro = np.NINF
        self.best_train_f1_macro = np.NINF
        self.best_train_loss = np.inf


    def forward(self, x):
        return self.model(x)
    '''
    # added for problem on gradient, anomaly detection in windows - gissella
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")
    '''
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = self.criterion(z, y)
        # accumulate and return metrics for logging
        acc = self.train_acc(z, y)
        f1_micro = self.train_f1_micro(z, y)
        f1_macro = self.train_f1_macro(z, y)
        #print("TRAIN",z,y)
        # just accumulate
        #self.train_auroc.update(z, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        self.log("train_f1_micro", f1_micro)
        self.log("train_f1_macro", f1_macro)
        #acc = self.accuracy(z, y)
        #print("Train Loss:",loss,"Train Acc:",acc)
        #self.log('train_loss', loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log('train_accuracy',acc ,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = self.criterion(z, y)
        #print("VAL",z,y)
        self.val_acc.update(z, y)
        self.val_f1_micro.update(z, y)
        self.val_f1_macro.update(z, y)
        #self.val_auroc.update(z, y)
        #acc = self.accuracy(z, y)
        #print("Val Loss:",loss, "Val Acc:", acc)
        #self.log('val_loss', loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_accuracy',acc ,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        #print('training steps', training_step_outputs)

        loss_acum = []

        for value in training_step_outputs:
            loss = value['loss']
            loss_acum.append(loss)

        avg_loss = sum(loss_acum)/len(loss_acum)

        self.epochCount = self.epochCount + 1

        # compute metrics
        train_accuracy = self.train_acc.compute()
        train_f1_micro = self.train_f1_micro.compute()
        train_f1_macro = self.train_f1_macro.compute()
        #train_auroc = self.train_auroc.compute()
        #print("TRAIN_E_E:",train_auroc)
        # log metrics
        self.log("train_accuracy", train_accuracy)
        self.log("train_f1_micro", train_f1_micro)
        self.log("train_f1_macro", train_f1_macro)
        # reset all metrics
        self.train_acc.reset()
        self.train_f1_micro.reset()
        self.train_f1_macro.reset()
        print(f"\ntraining accuracy: {train_accuracy:.4}, "\
        f"f1_micro: {train_f1_micro:.4}, "\
        f"f1_macro: {train_f1_macro:.4}") #\
        #f"auroc: {train_auroc:.4}")
        #result = pl.EvalResult(checkpoint_on=avg_loss)
        print("Train Lss AVR:",avg_loss)
        print("-------------------------------------------------------------")
        #result.log('train_avr_loss', avg_loss, on_epoch=True, prog_bar=True)

        if self.best_train_acc < train_accuracy:
            self.best_train_acc = train_accuracy
        if self.best_train_f1_micro < train_f1_micro:
            self.best_train_f1_micro = train_f1_micro
        if self.best_train_f1_macro < train_f1_macro:
            self.best_train_f1_macro = train_f1_macro
        if self.best_train_loss > avg_loss:
            self.best_train_loss = avg_loss

    def validation_epoch_end(self, validation_step_outputs):

        loss_acum = []

        for value in validation_step_outputs:
            loss = value
            loss_acum.append(loss)

        avg_loss = sum(loss_acum)/len(loss_acum)
        print("Epoch:", self.epochCount)
        print("Test Lss AVR:",avg_loss)

        # compute metrics
        val_accuracy = self.val_acc.compute()
        val_f1_micro = self.val_f1_micro.compute()
        val_f1_macro = self.val_f1_macro.compute()
        #val_auroc = self.val_auroc.compute()
        #print("VAL_E_END",val_auroc)
        # log metrics 
        self.log("val_accuracy", val_accuracy)
        self.log("val_f1_micro", val_f1_micro)
        self.log("val_f1_macro", val_f1_macro)
        #self.log("val_auroc", val_auroc)
        # reset all metrics
        self.val_acc.reset()
        self.val_f1_micro.reset()
        self.val_f1_macro.reset()
        #self.val_auroc.reset()
        print(f"\ntraining accuracy: {val_accuracy:.4}, "\
        f"f1_micro: {val_f1_micro:.4}, "\
        f"f1_macro: {val_f1_macro:.4}") #\
        #f"f1: {val_f1:.4}, auroc: {val_auroc:.4}")

        if self.best_val_acc < val_accuracy:
            self.best_val_acc = val_accuracy
        if self.best_val_f1_micro < val_f1_micro:
            self.best_val_f1_micro = val_f1_micro
        if self.best_val_f1_macro < val_f1_macro:
            self.best_val_f1_macro = val_f1_macro
        if self.best_val_loss > avg_loss:
            self.best_val_loss = avg_loss

    def on_train_end(self):
         print(f"\nbest val acc: {self.best_val_acc:.4}")
         print(f"\nbest val f1 micro: {self.best_val_f1_micro:.4}")
         print(f"\nbest val f1 macro: {self.best_val_f1_macro:.4}")
         print(f"\nbest val loss: {self.best_val_loss:.4}")
         print(f"\nbest train acc: {self.best_train_acc:.4}")
         print(f"\nbest train f1 micro: {self.best_train_f1_micro:.4}")
         print(f"\nbest train f1 macro: {self.best_train_f1_macro:.4}")
         print(f"\nbest train loss: {self.best_train_loss:.4}")

         dfData = pd.DataFrame({
             "version": [self.hparams.version],
             "ListWords": [str(self.hparams.num_classes) + " words"],
             "SequenceLen": [self.hparams.sequence_length],
             "LearningRate":[self.hparams.learning_rate],
             "Stride":[self.hparams.temporal_stride],
             "batch_size":[self.hparams.batch_size],
             "accumulated_batch_size":[self.hparams.accumulate_grad_batches],
             "DropOut":[self.hparams.dropout],
             "BestValAcc": [self.best_val_acc.cpu().data.numpy()],
             "BestValF1-Micro": [self.best_val_f1_micro.cpu().data.numpy()],
             "BestValF1-Macro": [self.best_val_f1_macro.cpu().data.numpy()],
             "BestValLoss": [self.best_val_loss.cpu().data.numpy()],
             "BestTrainAcc": [self.best_train_acc.cpu().data.numpy()],
             "BestTrainF1-Micro": [self.best_train_f1_micro.cpu().data.numpy()],
             "BestTrainF1-Macro": [self.best_train_f1_macro.cpu().data.numpy()],
             "BestTrainLoss": [self.best_train_loss.cpu().data.numpy()],
             "TestAcc":[np.NINF],
             "F1-micro":[np.NINF],
             "F1-macro":[np.NINF],
             "seed":[self.hparams.seed],
             "experiment-name":[self.hparams.csv_name]})

         if os.path.isfile(self.hparams.csv_name):
             df = pd.read_csv(self.hparams.csv_name,index_col=0)
             df = df.append(dfData)
             df.to_csv(self.hparams.csv_name)
             print ("%s updated"%(self.hparams.csv_name))
         else:
             df = pd.DataFrame()
             df = df.append(dfData)
             df.to_csv(self.hparams.csv_name)
             print ("File not exist - %s created"%(self.hparams.csv_name))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=0.1),
            'monitor': 'val_accuracy'
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        #parser.add_argument('--sequence_length', type=int, default=16)
        #parser.add_argument('--temporal_stride', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--cnn', type=str, default='rn18')
        #parser.add_argument('--batch_size', type=int, default=32)
        #parser.add_argument('--accumulate_grad_batches', type=int, default=8)
        parser.add_argument('--freeze_layers', type=int, default=0,
                            help='Freeze all CNN layers up to this index (default: 0, no frozen layers)')
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--dropout', help='Dropout before MHA and FC', type=float, default=0)
        parser.add_argument('--lr_step_size', type=int, default=5)
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--num_classes', type=int, default=10)
        return parser
