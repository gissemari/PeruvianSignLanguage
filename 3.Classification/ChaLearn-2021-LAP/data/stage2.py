import pandas as pd
from collections import Counter
import sys

#if train
print(sys.argv)
if sys.argv[1] == '1':

    print("Training in stage2")

    train_reader = pd.read_csv("train_labels.csv", encoding='utf-8', header=None)
    val_reader = pd.read_csv("val_labels.csv", encoding='utf-8', header=None)

    train_stage = [[name,label,'train'] for name, label in train_reader.values.tolist()]
    val_stage = [[name,label,'val'] for name, label in val_reader.values.tolist()]

    stage = train_stage + val_stage
    df = pd.DataFrame(stage)
    df.to_csv('train_val_labels_STAGE2.csv',index=False, header=False)

    # To check if there are some reapetead names
    train_name = [name for name, label in train_reader.values.tolist()]
    val_name = [name for name, label in val_reader.values.tolist()]

    trainRepeated = Counter(train_name)
    valRepeated = Counter(val_name)

    print("Train")
    print(trainRepeated)
    print("Val")
    print(valRepeated)


else:
    test_reader = pd.read_csv("test_labels.csv", encoding='utf-8',header=None)

    test_stage = [[name,label,'test'] for name, label in test_reader.values.tolist()]

    df = pd.DataFrame(test_stage)
    print(df)
    df.to_csv('test_labels_STAGE2.csv',index=False, header=False)
