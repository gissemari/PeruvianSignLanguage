"""PyTorch Lightning module definition.
Delegates computation to one of the defined networks (vtn.py, vtn_hc.py, vtn_hcpf.py)"""
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR

from .vtn_hcpf import VTNHCPF
from .vtn_hcpf_d import VTNHCPFD
import torchmetrics

def get_model_def():
    return Module


def get_model(**kwargs):
    return Module(**kwargs)


class Module(pl.LightningModule):

    def __init__(self, model, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        NUM_CLASSES = 10

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
        
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

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
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        acc = self.accuracy(z, y)
        #print("Train Loss:",loss,"Train Acc:",acc)
        self.log('train_loss', loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy',acc ,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        acc = self.accuracy(z, y)
        #print("Val Loss:",loss, "Val Acc:", acc)
        self.log('val_loss', loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy',acc ,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        #print('training steps', training_step_outputs)

        loss_acum = []

        for value in training_step_outputs:
            loss = value['loss']
            loss_acum.append(loss)

        avg_loss = sum(loss_acum)/len(loss_acum)
        #result = pl.EvalResult(checkpoint_on=avg_loss)
        print("Train Lss AVR:",avg_loss)
        print("-------------------------------------------------------------")
        #result.log('train_avr_loss', avg_loss, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):

        loss_acum = []

        for value in validation_step_outputs:
            loss = value
            loss_acum.append(loss)

        avg_loss = sum(loss_acum)/len(loss_acum)

        print("Test Lss AVR:",avg_loss)

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
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--embed_size', type=int, default=512)
        parser.add_argument('--cnn', type=str, default='rn18')
        parser.add_argument('--freeze_layers', type=int, default=0,
                            help='Freeze all CNN layers up to this index (default: 0, no frozen layers)')
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--dropout', help='Dropout before MHA and FC', type=float, default=0)
        parser.add_argument('--lr_step_size', type=int, default=5)
        parser.add_argument('--model', type=str, required=True)
        return parser
