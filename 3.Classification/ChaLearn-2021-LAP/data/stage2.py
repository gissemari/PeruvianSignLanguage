import pandas as pd
from collections import Counter

train_reader = pd.read_csv("train_labels.csv", encoding='utf-8')
val_reader = pd.read_csv("val_labels.csv", encoding='utf-8')

train_stage = [[name,label,'train'] for name, label in train_reader.values.tolist()]
val_stage = [[name,label,'val'] for name, label in val_reader.values.tolist()]

train_name = [name for name, label in train_reader.values.tolist()]
val_name = [name for name, label in val_reader.values.tolist()]

trainRepeated = Counter(train_name)
valRepeated = Counter(val_name)

print("Train")
print(trainRepeated)
print("Val")
print(valRepeated)

stage = train_stage + val_stage

df = pd.DataFrame(stage)

df.to_csv('train_val_labels_STAGE2.csv',index=False, header=False)
