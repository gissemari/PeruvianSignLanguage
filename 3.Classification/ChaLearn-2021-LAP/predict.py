"""Use a trained neural network to predict on a data set."""
import csv
import importlib
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from models import module

import pandas as pd

if __name__ == '__main__':
    # -------------------------------- #
    # ARGUMENT PARSING
    # -------------------------------- #
    parser = ArgumentParser()

    # Program specific
    parser.add_argument('--log_dir', type=str, help='Directory to which experiment logs will be written', required=True)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--dataset', type=str, help='Dataset module', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint', required=True)
    parser.add_argument('--submission_template', type=str, help='Path to the submission template', required=True)
    parser.add_argument('--out', type=str, help='Output file path', required=True)
    parser.add_argument('--subject', type=str, help='file of subjects', required=True)

    program_args, _ = parser.parse_known_args()

    # Model specific
    parser = module.get_model_def().add_model_specific_args(parser)

    # Data module specific
    data_module = importlib.import_module(f'datasets.{program_args.dataset}')
    parser = data_module.get_datamodule_def().add_datamodule_specific_args(parser)

    # Trainer specific
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # -------------------------------- #
    # SETUP
    # -------------------------------- #
    pl.seed_everything(args.seed)

    dict_args = vars(args)

    model = module.get_model_def().load_from_checkpoint(args.checkpoint)
    dm = data_module.get_datamodule(**dict_args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    # -------------------------------- #
    # USING MODEL TO PREDICT
    # -------------------------------- #
    submission = dict()

    dataloader = dm.test_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, paths = batch
            if isinstance(x, list):
                logits = model([e.to(device) for e in x]).cpu()
            else:
                logits = model(x.to(device)).cpu()

            predictions = torch.argmax(logits, dim=1)
            for j in range(logits.size(0)):
                submission[paths[j]] = predictions[j].item()
    #print(submission)
    with open(args.submission_template) as stf:
        reader = csv.reader(stf)
        totalRows = 0#len(list(reader))
        with open(args.out, 'w') as of:
            writer = csv.writer(of)
            accum = 0
            for row in reader:
                sample = row[0]
                #print(f'Predicting {sample}', end=' ')
                #print(f'as {submission[sample]} - pred {submission[sample]} and real {row[1]}')
                match=0
                if int(row[1]) == int(submission[sample]):
                    match=1
                    accum+=1
                totalRows+=1
                
                # identifying subject
                with open(args.subject) as subjectFile:
                    readerSubject = csv.reader(subjectFile)
                    idx = int(sample.split('_')[-1])
                    subjectName = 'NA'
                    for name, idxStart, idxEnd in readerSubject:
                        if (int(idxStart) <= idx) and (idx<= int(idxEnd)):
                            subjectName = name
                            break
                writer.writerow([sample, submission[sample], str(row[1]), str(match), subjectName])

    df = pd.read_csv("trainSummary_2.csv",index_col=0)
    df.loc[(df["SequenceLen"]==args.sequence_length) & (df["Stride"]==args.temporal_stride) & (df["LearningRate"]==args.learning_rate) & (df["seed"]==program_args.seed) ,["TestAcc"]] = accum/totalRows
    df.to_csv("trainSummary_2.csv")
    print(f'Accuracy for Test set {accum/totalRows}')
    print(f'Wrote submission to {args.out}')
