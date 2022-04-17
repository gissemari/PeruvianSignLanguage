import pandas as pd

def getWords(dict):

    words = []
    for pos, gloss in enumerate(dict):
        words.append((pos, dict[pos]["gloss"].upper()))

    return words

aec = pd.read_json('../Data/AEC/dict.json', encoding="utf-8")
pucp = pd.read_json('../Data/PUCP_PSL_DGI156/dict.json', encoding="utf-8")

aecWords = getWords(aec)
pucpWords = getWords(pucp)

similarWords = {repeted:{'aec':len(aec[aecPos]["instances"]),
                          'pucp':len(pucp[pucpPos]["instances"])}
                        for (aecPos, repeted) in aecWords 
                        for (pucpPos, pucpRepeted) in pucpWords 
                        if repeted == pucpRepeted} 

df = pd.DataFrame(data=similarWords) 
df= df.T

df.to_csv('repeatedWords.csv', encoding="utf-8")