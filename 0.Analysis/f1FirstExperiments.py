import sklearn
import pandas as pd 
import torchmetrics
import torch
import glob

preds5 = glob.glob("./../3.Classification/ChaLearn-2021-LAP/predictions/predsChameleon/*.csv")
#print(preds5)
#dfGround5 = pd.read_csv("./../3.Classification/ChaLearn-2021-LAP/Data_project_bckp/list5/test_labels.csv", header = None, encoding="utf-8")
#testListLabels = dfGround5.iloc[:,1]
for fileName in preds5:
    if int(fileName[-6:-4])<48:
        numClass = 5
    else:
        numClass = 10
    try:
        dfPreds = pd.read_csv(fileName, header=None)

        #print(dfPreds.iloc[:,1])
        #print(dfPreds.iloc[:,2])
        predLabels = torch.tensor(list(dfPreds.iloc[:,1]))
        groundLabels = torch.tensor(list(dfPreds.iloc[:,2]))
        #print(len(predLabels), len(groundLabels))
        #print(predLabels, groundLabels)

    except:
        try:
            dfPreds = pd.read_csv(fileName)
            predLabels = torch.tensor(list(dfPreds.iloc[:,1]))
            groundLabels = torch.tensor(list(dfPreds.iloc[:,2]))
        except:
            continue

    f1ScoreMacro = torchmetrics.F1(average='macro', num_classes=numClass)
    f1ScoreMicro = torchmetrics.F1(average='micro')
    print(fileName[-6:], f1ScoreMacro(predLabels,groundLabels), f1ScoreMicro(predLabels,groundLabels))