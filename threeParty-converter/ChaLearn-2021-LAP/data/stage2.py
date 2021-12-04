import pandas as pd

train_reader = pd.read_csv("train_labels.csv", encoding='utf-8')
val_reader = pd.read_csv("val_labels.csv", encoding='utf-8')

train_stage = [[name,label,'train'] for name, label in train_reader.values.tolist()]
val_stage = [[name,label,'val'] for name, label in val_reader.values.tolist()]

stage = train_stage + val_stage

df = pd.DataFrame(stage)

df.to_csv('train_val_labels_STAGE2.csv',index=False, header=False)
