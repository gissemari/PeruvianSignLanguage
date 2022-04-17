import pandas as pd
from collections import Counter

def getWordsPk(cleaned):
    data, labels, names = list(cleaned.keys())

    words = []
    for pos in range(len(cleaned[names])):
        label = cleaned[labels][pos]
        words.append((pos, label.upper()))
    return words

def getWords(dict):

    words = []
    for pos, gloss in enumerate(dict):
        words.append((pos, dict[pos]["gloss"].upper()))

    return words

cleaned = pd.read_pickle('../Data/AEC_cleaned/data_10_10_27.pk')
pucp = pd.read_json('../Data/PUCP_PSL_DGI156/dict.json', encoding="utf-8")

cleanedWords = getWordsPk(cleaned)
pucpWords = getWords(pucp)

data, labels, names = list(cleaned.keys())
labelCounter = Counter(cleaned[labels])

uppercaseLabelCounter = {}
for key in labelCounter:
    uppercaseLabelCounter[key.upper()] = labelCounter[key]

similarWords = {repeted:{'cleaned':uppercaseLabelCounter[repeted],
                          'pucp':len(pucp[pucpPos]["instances"])}
                        for (cleanedPos, repeted) in cleanedWords 
                        for (pucpPos, pucpRepeted) in pucpWords 
                        if repeted == pucpRepeted} 

df = pd.DataFrame(data=similarWords) 
df= df.T

df.to_csv('repeatedWords-cleaned-PUCP.csv', encoding="utf-8")